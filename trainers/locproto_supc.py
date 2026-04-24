import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from utils.trainer import TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import AverageMeter

from clip_w_local import clip_clear as clip
from clip_w_local.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
from tqdm import tqdm
from PIL import Image
from .zsclip_contra import entropy_select_topk2, CUSTOM_TEMPLATES
import os
import json
from copy import deepcopy
from utils.bonder import CrossAttnBlock
from utils.loss import SupConLoss

_tokenizer = _Tokenizer()
softmax = nn.Softmax(dim=1).cuda()


def entropy_select_topk(p, top_k, label, num_of_local_feature):
    label_repeat = label.repeat_interleave(num_of_local_feature)
    p = F.softmax(p, dim=-1)
    pred_topk = torch.topk(p, k=top_k, dim=1)[1]
    contains_label = pred_topk.eq(torch.tensor(label_repeat).unsqueeze(1)).any(dim=1)
    selected_p = p[~contains_label]

    if selected_p.shape[0] == 0:
        return torch.tensor([0]).cuda()
    return -torch.mean(torch.sum(selected_p * torch.log(selected_p+1e-5), 1))


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


def get_dense_logits2(image_features, local_image_features, all_text_features, mean_text_features, topk=50):
    base_logits = image_features @ mean_text_features.T   
    image_features = image_features.unsqueeze(1)  
    all_image_features = local_image_features
    w = torch.einsum('bmd,bnd->bmn', image_features, all_image_features) 

    mean_text_features = mean_text_features.unsqueeze(0) 
    _,n_cls,d = mean_text_features.shape
    all_text_features = all_text_features.reshape(-1, n_cls, d)
    v = torch.einsum('mcd,ncd->mnc', mean_text_features, all_text_features)  
    v = F.softmax(v, dim=1)
    sim = torch.einsum('bmd,ncd->bcmn', all_image_features, all_text_features)  
    sim, idx = sim.topk(dim=2, k=topk)    
    idx = idx[:, 0, :, 0].unsqueeze(1)
    w = torch.gather(w, dim=2, index=idx)
    w = F.softmax(w, dim=-1)
    weight = torch.einsum('bdm,dnc->bcmn', w,v) 
    mat = sim * weight
    
    bias_logits = torch.sum(mat, dim=(-2,-1))
    logits = base_logits + bias_logits
    return logits


def get_supc_loss(g_img_feats, id_loc_feats, ood_loc_feats, text_stu, text_tea, label, n_class=99, topk=50):
    bs, k, d = id_loc_feats.shape
    _, n_disc, _ = text_tea.shape
    id_ex_label = label.unsqueeze(1).repeat(1, k)
    ood_ex_label = torch.full((bs,), n_class).cuda()
    text_ex_label = label.unsqueeze(1).repeat(1, n_disc)

    features = torch.cat([id_loc_feats, ood_loc_feats], dim=0)
    res_label = torch.cat([label, ood_ex_label], dim=0)

    loss = SupConLoss(temperature=0.5, base_temperature=0.5)(features=features, labels=res_label)
    return loss


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        
        self.device = torch.device("cuda")
        clip_model.to(self.device)
        self.image_encoder = clip_model.visual
        self.zs_img_encoder = deepcopy(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.image_features_store =[]
        self.cfg = cfg

        description_file = os.path.join('./description', f'{cfg.DATASET.NAME}.json')
        print(f'Using description file: {description_file}')
        llm_descriptions = json.load(open(description_file))
        text_features =[]
        template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        all_prompt =[]
        print(classnames)
        for classname in classnames:
            prompts =[]
            prompt = template.format(classname.replace("_", " "))
            prompts.append(prompt)

            for i in range(50):
                prompt_desc = prompt + ' ' + llm_descriptions[classname.replace("_", " ")][i]
                prompts.append(prompt_desc)
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
            all_prompt.append(prompts)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    text_features.append(clip_model.encode_text(prompts)) 
                    
        self.all_prompt = torch.cat(all_prompt)

        text_features = torch.cat(text_features) 
        _, d = text_features.shape
        self.ndisc = 51
        text_features = text_features.view(self.ndisc, -1, d)
        self.all_text_features_tea = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_mean = text_features.mean(dim=0)
        self.text_features_tea = text_features_mean / text_features_mean.norm(dim=-1, keepdim=True)
        self.text_prototypes = self.all_text_features_tea   

        if cfg.is_bonder:
            self.bonder = CrossAttnBlock(512)
            self.bonder.to(self.dtype)

        # ---------------- TIP-ADAPTER-F PROPERTIES ----------------
        self.use_tip_adapter = False
        self.tip_adapter = None
        self.cache_values = None
        self.tip_alpha = 1.0  
        self.tip_beta = 5.5   

    def inject_tip_adapter(self, cache_keys, cache_values, n_cls):
        """Dynamically attach the cache after Stage 1 completes."""
        self.tip_adapter = nn.Linear(cache_keys.shape[1], cache_keys.shape[0], bias=False).to(self.dtype).cuda()
        self.tip_adapter.weight = nn.Parameter(cache_keys) 
        self.cache_values = cache_values.to(self.dtype).cuda()
        
        # Learnable Sigmoid Gate and Beta
        self.tip_alpha = nn.Parameter(torch.zeros(1, n_cls, dtype=self.dtype, device=self.device))
        self.tip_beta = nn.Parameter(torch.tensor(5.5, dtype=self.dtype, device=self.device))
        
        self.use_tip_adapter = True

    def forward(self, image, mask=None, labels=None, phase='glali'):
        
        # If we are in Stage 2 (tip_train) or Inference, WE DO NOT TRACK GRADIENTS for GLAli
        # This saves massive amounts of VRAM!
        context_manager = torch.no_grad() if phase in ['tip_train', 'eval'] else torch.enable_grad()
        
        with context_manager:
            with torch.no_grad(): # Teacher is always frozen
                image_features_tea, local_image_features_tea, _ = self.zs_img_encoder(image.to(self.dtype))
                image_features_tea = image_features_tea / image_features_tea.norm(dim=-1, keepdim=True)
            
            image_features, local_image_features, _  = self.image_encoder(image.to(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)

            text_prototypes = self.text_prototypes.detach()
            n_disc, c, d = text_prototypes.shape
            id_loc_feats = None
            ood_loc_feats = None
            l2p = None
            l2p_tea = None
            
            if labels is not None and self.cfg.is_bonder:
                bs = labels.shape[0]
                l2p = text_prototypes[torch.arange(n_disc).view(-1, 1).expand(n_disc, bs), labels, :]
                l2p_tea = self.all_text_features_tea[torch.arange(n_disc).view(-1, 1).expand(n_disc, bs), labels, :]
                l2p = torch.transpose(l2p, 0, 1)
                l2p_tea = torch.transpose(l2p_tea, 0, 1)

                sim = local_image_features @ (l2p.mean(dim=1, keepdim=True).transpose(1,2))
                sim = sim.squeeze(-1)
                _, idx = torch.topk(input=sim, k=self.cfg.topk)
                _, idx_ood = torch.topk(input=sim, k=self.cfg.topk, largest=False)

                l2p_loc = l2p[:, 1:, :]
                selected_loc_img_feats = torch.gather(local_image_features, 1, idx.unsqueeze(-1).expand(-1, -1, d))

                id_loc_feats = selected_loc_img_feats
                ood_loc_feats = torch.gather(local_image_features, 1, idx_ood.unsqueeze(-1).expand(-1, -1, d))
                
                text_bias = self.bonder(l2p_loc, selected_loc_img_feats.detach())
                text_bias = text_bias / text_bias.norm(dim=-1, keepdim=True)
                alpha = self.cfg.lambda_value
                updated_proto = self.text_prototypes
                
                contra_labels = torch.arange(c).view(-1,1).cuda()
                mask = torch.eq(labels.unsqueeze(1), contra_labels.T).to(self.dtype).cuda()
                update_features = torch.matmul(mask.view(bs, c).transpose(0,1).unsqueeze(0).repeat(n_disc-1,1,1), text_bias.transpose(1, 0))
                proto_mask = torch.zeros(c, dtype=torch.int).cuda()
                proto_mask[labels] = 1
                proto_mask = proto_mask.view(1, -1, 1).repeat(n_disc, 1, d)
                update_features = torch.cat([self.text_prototypes[0:1, :, :], update_features], dim=0)
                updated_proto = (1-proto_mask) * updated_proto + proto_mask * (alpha * updated_proto + (1-alpha) * update_features)

                updated_proto_norm = updated_proto / updated_proto.norm(dim=-1, keepdim=True)
                updated_proto_mean = updated_proto_norm.mean(dim=0)
                updated_proto_mean_norm = updated_proto_mean / updated_proto_mean.norm(dim=-1, keepdim=True)
            else:
                updated_proto_norm = self.text_prototypes / self.text_prototypes.norm(dim=-1, keepdim=True)
                updated_proto_mean = updated_proto_norm.mean(dim=0)
                updated_proto_mean_norm = updated_proto_mean / updated_proto_mean.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            
            logits = logit_scale * get_dense_logits2(image_features.detach(), local_image_features.detach(), updated_proto_norm, updated_proto_mean_norm, topk=self.cfg.topk)
            logits_local = logit_scale * get_dense_logits2(image_features, local_image_features, self.all_text_features_tea.detach(), self.text_features_tea.detach(), topk=self.cfg.topk)

        # ---------------- STAGE 2 & INFERENCE: LOCAL LESION TIP-ADAPTER ----------------
        if self.use_tip_adapter:
            bs = image.shape[0]
            if labels is not None and phase == 'tip_train':
                # Use GT labels to extract lesions exactly like we did when building the cache
                l2p_eval = updated_proto_norm[torch.arange(n_disc).view(-1, 1).expand(n_disc, bs), labels, :]
                l2p_eval = torch.transpose(l2p_eval, 0, 1)
                sim_eval = local_image_features @ (l2p_eval.mean(dim=1, keepdim=True).transpose(1, 2))
                sim_eval = sim_eval.squeeze(-1)
            else:
                # Inference: Blindly find the most pathological patches
                text_protos_mean = updated_proto_mean_norm 
                sim_all = torch.matmul(local_image_features, text_protos_mean.T) 
                sim_eval, _ = torch.max(sim_all, dim=-1) 

            _, idx_lesion = torch.topk(sim_eval, k=self.cfg.topk, dim=1)
            lesion_patches = torch.gather(local_image_features, 1, idx_lesion.unsqueeze(-1).expand(-1, -1, d))
            lesion_query = lesion_patches.mean(dim=1)
            lesion_query = F.normalize(lesion_query, p=2, dim=-1)
            
            # NOTE: THIS PART IS TRACKED BY AUTOGRAD DURING tip_train
            affinity = self.tip_adapter(lesion_query)
            affinity = torch.clamp(affinity, max=1.0)
            
            # DTYPE SAFETY FIX
            safe_beta = F.softplus(self.tip_beta).to(affinity.dtype) if isinstance(self.tip_beta, torch.Tensor) else self.tip_beta
            gate = torch.sigmoid(self.tip_alpha).to(affinity.dtype) if isinstance(self.tip_alpha, torch.Tensor) else self.tip_alpha
            
            cache_logits = torch.exp(-safe_beta * (1.0 - affinity)) @ self.cache_values.to(affinity.dtype)
            
            scaled_cache_logits = (cache_logits * logit_scale) * gate
            
            tip_logits = logits + scaled_cache_logits
            tip_logits_local = logits_local + scaled_cache_logits
            
            if phase == 'tip_train':
                return {'tip_logits': tip_logits}
            else:
                logits = tip_logits
                logits_local = tip_logits_local
        # -------------------------------------------------------------------------------

        return logits, logits_local, image_features_tea, image_features, updated_proto_norm, id_loc_feats, ood_loc_feats, l2p, l2p_tea


@TRAINER_REGISTRY.register()
class LocProto(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.LOCOOP.PREC in["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        self.lambda_value = cfg.lambda_value
        self.top_k = cfg.topk
        self.label =[]

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.LOCOOP.PREC in["fp32", "amp"]:
            clip_model.float()

        print("Building pure GLAli model...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Configuring Gradients: Vision Encoder + Bonder")
        for name, param in self.model.named_parameters():
            if 'image_encoder.transformer.resblocks.11.attn' in name or 'bonder' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        self.model.to(self.device)
        
        if "ViT" in cfg.MODEL.BACKBONE.NAME:
            self.optim = build_optimizer(self.model.image_encoder.transformer.resblocks[-1].attn, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("attn_learner", self.model.image_encoder.transformer.resblocks[-1].attn, self.optim, self.sched)
            
            if cfg.is_bonder:
                cfg.OPTIM2 = deepcopy(cfg.OPTIM)
                cfg.OPTIM2.LR = cfg.OPTIM.LR
                self.optim2 = build_optimizer(self.model.bonder, cfg.OPTIM2)
                self.sched2 = build_lr_scheduler(self.optim2, cfg.OPTIM2)
                self.register_model("bonder_learner", self.model.bonder, self.optim2, self.sched2)

        self.scaler = GradScaler() if cfg.TRAINER.LOCOOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            self.model = nn.DataParallel(self.model)

    def train(self):
        """Override normal training to implement Two-Stage Sequence."""
        # --- STAGE 1: Train Pure GLAli ---
        print("\n" + "="*50)
        print("STAGE 1: Training Pure GLAli")
        print("="*50)
        # Call TrainerBase.train() empty parameters (SimpleTrainer handles start/max natively)
        super().train()
        
        # --- STAGE 2: Train Local Lesion Tip-Adapter-F ---
        print("\n" + "="*50)
        print("STAGE 2: Training Local Lesion Tip-Adapter-F")
        print("="*50)
        self.train_tip_adapter()
        
        print("Saving final text prototypes...")
        target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save(target_model.text_prototypes.detach().cpu(), osp.join(self.output_dir, 'proto.pth'))
        
        print("Deploying the final hybrid model")
        self.test()
        
    def after_train(self):
        # Suppress the default Dassl after_train testing because we moved it to the end of our Stage 2
        pass

    @torch.no_grad()
    def build_lesion_cache(self):
        print("Extracting Trained Visual Memory Cache (Lesion-Only)...")
        
        # FIX: Build a clean dataloader with test transforms and sequential sampling
        from utils.data_manager import build_data_loader
        from dassl.data.transforms import build_transform
        tfm_test = build_transform(self.cfg, is_train=False)
        cache_loader = build_data_loader(
            self.cfg,
            sampler_type="SequentialSampler",
            data_source=self.dm.dataset.train_x,
            batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False
        )

        cache_keys =[]
        cache_labels =[]
        
        target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        target_model.eval()
        
        text_protos = target_model.text_prototypes.detach()
        n_disc, c, d = text_protos.shape
        
        for batch in tqdm(cache_loader, desc="Building Cache"):
            image = batch["img"].to(self.device)
            label = batch["label"].to(self.device)
            bs = image.shape[0]
            
            _, local_feat, _ = target_model.image_encoder(image.type(target_model.dtype))
            local_feat = F.normalize(local_feat, p=2, dim=-1)
            
            l2p = text_protos[torch.arange(n_disc).view(-1, 1).expand(n_disc, bs), label, :]
            l2p = torch.transpose(l2p, 0, 1)
            
            sim = local_feat @ (l2p.mean(dim=1, keepdim=True).transpose(1, 2))
            sim = sim.squeeze(-1)
            
            _, idx = torch.topk(input=sim, k=self.cfg.topk)
            lesion_patches = torch.gather(local_feat, 1, idx.unsqueeze(-1).expand(-1, -1, d))
            lesion_query = lesion_patches.mean(dim=1)
            lesion_query = F.normalize(lesion_query, p=2, dim=-1)
            
            cache_keys.append(lesion_query.cpu())
            cache_labels.append(label.cpu())
            
        cache_keys = torch.cat(cache_keys, dim=0).to(self.device).to(target_model.dtype)
        cache_labels = torch.cat(cache_labels, dim=0).to(self.device)
        cache_values = F.one_hot(cache_labels, num_classes=self.num_classes).float().to(self.device).to(target_model.dtype)
        
        return cache_keys, cache_values

    def train_tip_adapter(self):
        target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        cache_keys, cache_values = self.build_lesion_cache()
        target_model.inject_tip_adapter(cache_keys, cache_values, len(self.dm.dataset.classnames))
        
        print("Configuring Tip-Adapter-F Optimizer...")
        tip_params =[
            target_model.tip_adapter.weight,
            target_model.tip_alpha,
            target_model.tip_beta
        ]
        optimizer_tip = torch.optim.AdamW(tip_params, lr=0.001, eps=1e-4)
        
        train_epochs_tip = 20
        scheduler_tip = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_tip, train_epochs_tip * len(self.train_loader_x))
        scaler_tip = GradScaler() if self.cfg.TRAINER.LOCOOP.PREC == "amp" else None
        
        for epoch in range(train_epochs_tip):
            self.model.train()
            if isinstance(self.model, nn.DataParallel):
                self.model.module.image_encoder.eval()
                self.model.module.text_encoder.eval()
                self.model.module.bonder.eval()
                self.model.module.zs_img_encoder.eval()
            else:
                self.model.image_encoder.eval()
                self.model.text_encoder.eval()
                self.model.bonder.eval()
                self.model.zs_img_encoder.eval()
                
            losses = AverageMeter()
            accs = AverageMeter()
            
            for batch_idx, batch in enumerate(tqdm(self.train_loader_x, desc=f"Tip-Adapter Epoch {epoch+1}/{train_epochs_tip}")):
                image = batch["img"].to(self.device)
                label = batch["label"].to(self.device)
                
                with autocast() if self.cfg.TRAINER.LOCOOP.PREC == "amp" else torch.enable_grad():
                    logits_dict = self.model(image, labels=label, phase='tip_train')
                    tip_logits = logits_dict['tip_logits']
                    loss = F.cross_entropy(tip_logits, label)
                    
                optimizer_tip.zero_grad()
                if scaler_tip is not None:
                    scaler_tip.scale(loss).backward()
                    scaler_tip.step(optimizer_tip)
                    scaler_tip.update()
                else:
                    loss.backward()
                    optimizer_tip.step()
                
                scheduler_tip.step()
                
                acc = compute_accuracy(tip_logits, label)[0].item()
                losses.update(loss.item(), image.size(0))
                accs.update(acc, image.size(0))
                
            print(f"Tip-Adapter-F Epoch {epoch+1}/{train_epochs_tip} - Loss: {losses.avg:.4f}, Acc: {accs.avg:.2f}%")
            
        print("--- Tip-Adapter-F Training Complete ---")
        
        # Explicitly save Tip-Adapter specific state variables for evaluation!
        tip_state = {
            "tip_adapter.weight": target_model.tip_adapter.weight.detach().cpu(),
            "cache_values": target_model.cache_values.detach().cpu(),
            "tip_alpha": target_model.tip_alpha.detach().cpu(),
            "tip_beta": target_model.tip_beta.detach().cpu()
        }
        torch.save(tip_state, osp.join(self.output_dir, "tip_state.pth"))
        print(f"Saved Tip-Adapter state to {osp.join(self.output_dir, 'tip_state.pth')}")

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.LOCOOP.PREC

        if prec == "amp":
            with autocast():
                output, output_local, img_feat_tea, img_feat_stu, text_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea = self.model(image, labels=label)
                all_text_features_tea = self.model.all_text_features_tea.clone()
                
                loss_id = F.cross_entropy(output, label)
                loss_id2 = F.cross_entropy(output_local, label)
                loss_distil_img = F.l1_loss(img_feat_tea, img_feat_stu, reduction='mean') * 10
                loss_distil_text = F.l1_loss(all_text_features_tea, text_stu, reduction='mean') * 25
                loss_supc = get_supc_loss(img_feat_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea, label, topk=self.top_k) * 0.5
                
                loss = loss_id + loss_id2 + loss_distil_img + loss_distil_text + loss_supc

            for name in self._optims:
                if self._optims[name] is not None:
                    self._optims[name].zero_grad()
                    
            self.scaler.scale(loss).backward()
            
            for name in self._optims:
                if self._optims[name] is not None:
                    self.scaler.step(self._optims[name])
                    
            self.scaler.update()
        else:
            output, output_local, img_feat_tea, img_feat_stu, text_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea = self.model(image, labels=label)
            all_text_features_tea = self.model.all_text_features_tea.clone()
            
            loss_id = F.cross_entropy(output, label)
            loss_id2 = F.cross_entropy(output_local, label)
            loss_distil_img = F.l1_loss(img_feat_tea, img_feat_stu, reduction='mean') * 10
            loss_distil_text = F.l1_loss(all_text_features_tea, text_stu, reduction='mean') * 25
            loss_supc = get_supc_loss(img_feat_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea, label, topk=self.top_k) * 0.5
            
            loss = loss_id + loss_id2 + loss_distil_img + loss_distil_text + loss_supc

            for name in self._optims:
                if self._optims[name] is not None:
                    self._optims[name].zero_grad()
                    
            loss.backward()
            
            for name in self._optims:
                if self._optims[name] is not None:
                    self._optims[name].step()

        loss_summary = {
            "loss": loss.item(),
            "loss_id": loss_id.item(),
            "loss_distil_img": loss_distil_img.item(),
            "loss_distil_text": loss_distil_text.item(),
            "acc": compute_accuracy(output_local, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        self.model.text_prototypes = text_stu.detach()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"

        # 1. Load the base GLAli model
        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                print(f"Warning: Model not found at {model_path}")
                continue

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]

            keys_to_delete =[k for k in state_dict.keys() if "token_prefix" in k or "token_suffix" in k]
            for k in keys_to_delete:
                del state_dict[k]

            print(f'Loading weights to {name} from "{model_path}"')
            self._models[name].load_state_dict(state_dict, strict=False)

        # ---------------- LOAD TIP-ADAPTER STATE IF EXISTS ----------------
        tip_state_path = osp.join(directory, "tip_state.pth")
        if osp.exists(tip_state_path):
            print(f"Found Tip-Adapter state at {tip_state_path}. Injecting into model...")
            tip_state = torch.load(tip_state_path, map_location="cpu")
            target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            
            weight = tip_state["tip_adapter.weight"]
            target_model.tip_adapter = nn.Linear(weight.shape[1], weight.shape[0], bias=False).to(target_model.dtype).cuda()
            target_model.tip_adapter.weight = nn.Parameter(weight.cuda())
            target_model.cache_values = tip_state["cache_values"].cuda()
            target_model.tip_alpha = nn.Parameter(tip_state["tip_alpha"].cuda())
            target_model.tip_beta = nn.Parameter(tip_state["tip_beta"].cuda())
            
            target_model.use_tip_adapter = True
            print("Tip-Adapter successfully restored for evaluation!")
        # ------------------------------------------------------------------

    @torch.no_grad()
    def test(self, split=None):
        target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        self.model.image_features_store =[]
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "test":
            split = "test"  
            data_loader = self.test_loader
        else:
            split = "train"
            data_loader = self.train_loader_x

        print(f"Evaluate on the *{split}* set")

        if self.cfg.is_bonder:
            proto_path = osp.join(self.output_dir, 'proto.pth')
            if osp.exists(proto_path):
                target_model.text_prototypes = torch.load(proto_path).to(self.device)
            else:
                print(f"Warning: {proto_path} not found. Using prototypes in memory.")
                
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            if len(output) >= 2:
                if self.cfg.is_bonder:
                    output = output[1] + 0.05 * output[0]
                else:
                    output = output[0]
            self.label.append(label)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def test_ood(self, data_loader, T):
        target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        self.model.image_features_store =[]
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.evaluator.reset()
        
        if self.cfg.is_bonder:
            proto_path = osp.join(self.output_dir, 'proto.pth')
            if osp.exists(proto_path):
                target_model.text_prototypes = torch.load(proto_path).to(self.device)

        mcm_score =[]
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            (images, labels, *id_flag) = batch
            if isinstance(images, str):
                images, label = self.parse_batch_test(batch)
            else:
                images = images.cuda()
            
            output, output_local, _, _, _, _, _, _, _ = self.model_inference(images)
            
            if self.cfg.is_bonder:
                output = output_local + 0.05 * output
                
            output /= 100.0
            smax_global = to_np(F.softmax(output/T, dim=-1))  
            mcm_global_score = -np.max(smax_global, axis=1)
            mcm_score.append(mcm_global_score)

        res = concat(mcm_score)[:len(data_loader.dataset)].copy()
        return res, res, res, res