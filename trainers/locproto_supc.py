import os.path as osp
import os
import json
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from PIL import Image

from dassl.engine import TRAINER_REGISTRY
from utils.trainer import TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip_w_local import clip_clear as clip
from clip_w_local.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .zsclip_contra import entropy_select_topk2, CUSTOM_TEMPLATES
from utils.bonder import CrossAttnBlock
from utils.loss import SupConLoss
from utils.data_manager import build_data_loader
from dassl.data.transforms import build_transform

_tokenizer = _Tokenizer()

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


def get_dense_logits2(image_features, local_image_features, all_text_features, mean_text_features, topk=50, global_weight=1.0):
    base_logits = image_features @ mean_text_features.T   
    image_features = image_features.unsqueeze(1)  
    all_image_features = local_image_features
    w = torch.einsum('bmd,bnd->bmn', image_features, all_image_features) 

    mean_text_features = mean_text_features.unsqueeze(0) 
    _, n_cls, d = mean_text_features.shape
    all_text_features = all_text_features.reshape(-1, n_cls, d)
    v = torch.einsum('mcd,ncd->mnc', mean_text_features, all_text_features)  
    v = F.softmax(v, dim=1)
    sim = torch.einsum('bmd,ncd->bcmn', all_image_features, all_text_features)  
    sim, idx = sim.topk(dim=2, k=topk)    
    idx = idx[:, 0, :, 0].unsqueeze(1)
    w = torch.gather(w, dim=2, index=idx)
    w = F.softmax(w, dim=-1)
    weight = torch.einsum('bdm,dnc->bcmn', w, v) 
    mat = sim * weight
    
    bias_logits = torch.sum(mat, dim=(-2, -1))
    
    # Applied Global Weight to the base logits to balance semantic pull
    logits = (global_weight * base_logits) + bias_logits
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


class TipAdapter(nn.Module):
    """
    Tweak 3: Encapsulates Tip-Adapter Parameters for clean AdamW optimization.
    """
    def __init__(self, cache_keys, n_cls, dtype):
        super().__init__()
        self.weight = nn.Parameter(cache_keys)
        # Tweak 3: Sigmoid gate alpha, tip beta, and global weight are parameters
        self.tip_alpha = nn.Parameter(torch.zeros(1, n_cls, dtype=dtype))
        self.tip_beta = nn.Parameter(torch.tensor(5.5, dtype=dtype))
        self.global_weight = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        
    def forward(self, x):
        return F.linear(x, self.weight)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, cache_keys=None, cache_values=None):
        super().__init__()
        self.device = torch.device("cuda")
        clip_model.to(self.device)
        self.image_encoder = clip_model.visual
        
        # GLAli Core: Frozen Teacher
        self.zs_img_encoder = deepcopy(clip_model.visual)
        for param in self.zs_img_encoder.parameters():
            param.requires_grad_(False)
            
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg

        description_file = os.path.join('./description', f'{cfg.DATASET.NAME}.json')
        print(f'Using description file: {description_file}')
        llm_descriptions = json.load(open(description_file))
        text_features = []
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
            prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
            all_prompt.append(prompts_tokenized)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    text_features.append(clip_model.encode_text(prompts_tokenized)) 
                    
        self.all_prompt = torch.cat(all_prompt)

        text_features = torch.cat(text_features) 
        _, d = text_features.shape
        self.ndisc = 51
        text_features = text_features.view(self.ndisc, -1, d)
        
        # Anti-NaN: Safe Normalization
        self.all_text_features_tea = F.normalize(text_features, p=2, dim=-1, eps=1e-5)
        text_features_mean = text_features.mean(dim=0)
        self.text_features_tea = F.normalize(text_features_mean, p=2, dim=-1, eps=1e-5)
        self.text_prototypes = self.all_text_features_tea   

        if cfg.is_bonder:
            self.bonder = CrossAttnBlock(512)
            self.bonder.to(self.dtype)

        self.tip_adapter = None
        if cache_keys is not None:
            print("Initializing Tip-Adapter Module...")
            self.tip_adapter = TipAdapter(cache_keys, len(classnames), self.dtype).to(self.device)
            self.register_buffer("cache_values", cache_values.to(self.dtype).to(self.device))

    def forward(self, image, mask=None, labels=None):
        # GLAli Core: Teacher
        with torch.no_grad():
            image_features_tea, local_image_features_tea, _ = self.zs_img_encoder(image.to(self.dtype))
            image_features_tea = F.normalize(image_features_tea, p=2, dim=-1, eps=1e-5)
            local_image_features_tea = F.normalize(local_image_features_tea, p=2, dim=-1, eps=1e-5)
        
        # GLAli Core: Student
        image_features, local_image_features, _  = self.image_encoder(image.to(self.dtype))
        image_features = F.normalize(image_features, p=2, dim=-1, eps=1e-5)
        local_image_features = F.normalize(local_image_features, p=2, dim=-1, eps=1e-5)

        text_prototypes = self.text_prototypes.detach()
        n_disc, c, d = text_prototypes.shape
        id_loc_feats = None
        ood_loc_feats = None
        l2p = None
        l2p_tea = None
        
        # GLAli Core: Bonder & LocSC Alignment
        if labels is not None and self.cfg.is_bonder:
            bs = labels.shape[0]
            l2p = text_prototypes[torch.arange(n_disc).view(-1, 1).expand(n_disc, bs), labels, :]
            l2p_tea = self.all_text_features_tea[torch.arange(n_disc).view(-1, 1).expand(n_disc, bs), labels, :]
            l2p = torch.transpose(l2p, 0, 1)
            l2p_tea = torch.transpose(l2p_tea, 0, 1)

            sim = local_image_features @ (l2p.mean(dim=1, keepdim=True).transpose(1, 2))
            sim = sim.squeeze(-1)
            _, idx = torch.topk(input=sim, k=self.cfg.topk)
            _, idx_ood = torch.topk(input=sim, k=self.cfg.topk, largest=False)

            l2p_loc = l2p[:, 1:, :]
            selected_loc_img_feats = torch.gather(local_image_features, 1, idx.unsqueeze(-1).expand(-1, -1, d))

            id_loc_feats = selected_loc_img_feats
            ood_loc_feats = torch.gather(local_image_features, 1, idx_ood.unsqueeze(-1).expand(-1, -1, d))
            
            text_bias = self.bonder(l2p_loc, selected_loc_img_feats.detach())
            text_bias = F.normalize(text_bias, p=2, dim=-1, eps=1e-5)
            
            alpha = self.cfg.lambda_value
            updated_proto = self.text_prototypes
            
            contra_labels = torch.arange(c).view(-1, 1).cuda()
            mask_labels = torch.eq(labels.unsqueeze(1), contra_labels.T).to(self.dtype).cuda()
            
            update_features = torch.matmul(mask_labels.view(bs, c).transpose(0, 1).unsqueeze(0).repeat(n_disc-1, 1, 1), text_bias.transpose(1, 0))
            
            proto_mask = torch.zeros(c, dtype=torch.int).cuda()
            proto_mask[labels] = 1
            proto_mask = proto_mask.view(1, -1, 1).repeat(n_disc, 1, d)
            
            update_features = torch.cat([self.text_prototypes[0:1, :, :], update_features], dim=0)
            updated_proto = (1 - proto_mask) * updated_proto + proto_mask * (alpha * updated_proto + (1 - alpha) * update_features)

            updated_proto_norm = F.normalize(updated_proto, p=2, dim=-1, eps=1e-5)
            updated_proto_mean = updated_proto_norm.mean(dim=0)
            updated_proto_mean_norm = F.normalize(updated_proto_mean, p=2, dim=-1, eps=1e-5)
        else:
            updated_proto_norm = F.normalize(self.text_prototypes, p=2, dim=-1, eps=1e-5)
            updated_proto_mean = updated_proto_norm.mean(dim=0)
            updated_proto_mean_norm = F.normalize(updated_proto_mean, p=2, dim=-1, eps=1e-5)

        # Anti-NaN: Safe logit scale clamp to prevent FP16 Overflow
        logit_scale = self.logit_scale.clamp(max=4.6051).exp()
        
        g_weight = self.tip_adapter.global_weight if self.tip_adapter is not None else 1.0
        
        logits = logit_scale * get_dense_logits2(image_features.detach(), local_image_features.detach(), updated_proto_norm, updated_proto_mean_norm, topk=self.cfg.topk, global_weight=g_weight)
        
        # Local Logits use the Teacher Text (Frozen) for stability
        logits_local = logit_scale * get_dense_logits2(image_features, local_image_features, self.all_text_features_tea.detach(), self.text_features_tea.detach(), topk=self.cfg.topk, global_weight=g_weight)

        # ---------------- HYBRID TIP-ADAPTER ----------------
        if self.tip_adapter is not None:
            # TWEAK 1: Domain Drift Fix. Query using frozen teacher features
            text_tea_frozen = self.text_features_tea.to(local_image_features_tea.device).to(self.dtype) 
            sim_to_all = torch.matmul(local_image_features_tea, text_tea_frozen.T)
            max_sim_per_patch, _ = torch.max(sim_to_all, dim=-1) 
            
            _, idx_lesion = torch.topk(max_sim_per_patch, k=self.cfg.topk, dim=1)
            
            lesion_patches = torch.gather(local_image_features_tea, 1, idx_lesion.unsqueeze(-1).expand(-1, -1, d))
            lesion_query_tea = F.normalize(lesion_patches.mean(dim=1), p=2, dim=-1, eps=1e-5)
            
            # TWEAK 2: Hybrid Query (Global + Lesion)
            hybrid_query = F.normalize(image_features_tea + lesion_query_tea, p=2, dim=-1, eps=1e-5)
            
            affinity = self.tip_adapter(hybrid_query)
            affinity = torch.clamp(affinity, max=1.0) # Prevent negative distances
            
            # Safe beta exponent
            safe_beta = torch.clamp(F.softplus(self.tip_adapter.tip_beta), min=0.0, max=20.0)
            
            cache_logits = torch.exp(-safe_beta * (1.0 - affinity)) @ self.cache_values.to(affinity.dtype)
            
            # TWEAK 3: Sigmoid Gating safely mapped to AMP precision
            gate = torch.sigmoid(self.tip_adapter.tip_alpha).to(affinity.dtype)
            
            # Scale cache up to CLIP mathematical space, safely gated
            scaled_cache_logits = (cache_logits * logit_scale) * gate
            
            logits = logits + scaled_cache_logits
            logits_local = logits_local + scaled_cache_logits

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

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.LOCOOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        # ---------------- TWEAK 2: CONSTRUCT HYBRID CACHE ----------------
        print("Extracting Pristine Visual Memory Cache (Hybrid) from training set...")
        tfm_test = build_transform(cfg, is_train=False)
        cache_loader = build_data_loader(
            cfg,
            sampler_type="SequentialSampler",
            data_source=self.dm.dataset.train_x,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False
        )

        clip_model.to(self.device)
        clip_model.eval()
        
        cache_keys = []
        cache_labels =[]
        
        with torch.no_grad():
            # Get text features explicitly to find lesion patches
            text_tea = clip_model.encode_text(clip.tokenize(["a photo of a " + c.replace("_", " ") for c in classnames]).to(self.device))
            text_tea = F.normalize(text_tea, p=2, dim=-1, eps=1e-5)
            
            for batch in tqdm(cache_loader, desc="Building Hybrid Cache"):
                image = batch["img"].to(self.device)
                label = batch["label"].to(self.device)
                
                img_feat, local_feat, _ = clip_model.visual(image.type(clip_model.dtype))
                img_feat = F.normalize(img_feat, p=2, dim=-1, eps=1e-5)
                local_feat = F.normalize(local_feat, p=2, dim=-1, eps=1e-5)
                
                gt_text = text_tea[label].unsqueeze(1) 
                sim_to_gt = torch.bmm(local_feat, gt_text.transpose(1, 2)).squeeze(-1)
                _, idx_pos = torch.topk(sim_to_gt, k=self.top_k, dim=1)
                
                d_dim = local_feat.shape[-1]
                lesion_feats = torch.gather(local_feat, 1, idx_pos.unsqueeze(-1).expand(-1, -1, d_dim))
                lesion_feat_mean = F.normalize(lesion_feats.mean(dim=1), p=2, dim=-1, eps=1e-5)
                
                # Hybrid Key: Global + Lesion
                hybrid_key = F.normalize(img_feat + lesion_feat_mean, p=2, dim=-1, eps=1e-5)
                
                cache_keys.append(hybrid_key.cpu())
                cache_labels.append(label.cpu())
                
        cache_keys = torch.cat(cache_keys, dim=0).to(self.device).to(clip_model.dtype) 
        cache_labels = torch.cat(cache_labels, dim=0).to(self.device) 
        cache_values = F.one_hot(cache_labels, num_classes=len(classnames)).float().to(self.device) 

        print("Building Custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model, cache_keys=cache_keys, cache_values=cache_values)

        print("Configuring Gradients: Vision Encoder + Bonder + Tip-Adapter")
        for name, param in self.model.named_parameters():
            if ('image_encoder.transformer.resblocks.11.attn' in name or 
                'bonder' in name or 
                'tip_adapter' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        self.model.to(self.device)
        
        if "ViT" in cfg.MODEL.BACKBONE.NAME:
            # Safe Learning Rate for Vision Encoder to prevent exploding loss_id
            cfg_vision = deepcopy(cfg.OPTIM)
            cfg_vision.LR = 1e-4 
            
            self.optim = build_optimizer(self.model.image_encoder.transformer.resblocks[-1].attn, cfg_vision)
            self.sched = build_lr_scheduler(self.optim, cfg_vision)
            self.register_model("attn_learner", self.model.image_encoder.transformer.resblocks[-1].attn, self.optim, self.sched)
            
            if cfg.is_bonder:
                cfg_bonder = deepcopy(cfg.OPTIM)
                cfg_bonder.LR = 1e-4
                self.optim2 = build_optimizer(self.model.bonder, cfg_bonder)
                self.sched2 = build_lr_scheduler(self.optim2, cfg_bonder)
                self.register_model("bonder_learner", self.model.bonder, self.optim2, self.sched2)

            # TWEAK 3: Hardcoded AdamW explicitly for Cache components
            if hasattr(self.model, "tip_adapter") and self.model.tip_adapter is not None:
                self.optim_tip = torch.optim.AdamW(self.model.tip_adapter.parameters(), lr=0.001, weight_decay=1e-4)
                cfg_tip = deepcopy(cfg.OPTIM)
                cfg_tip.LR = 0.001
                self.sched_tip = build_lr_scheduler(self.optim_tip, cfg_tip)
                self.register_model("tip_adapter_learner", self.model.tip_adapter, self.optim_tip, self.sched_tip)

        self.scaler = GradScaler() if cfg.TRAINER.LOCOOP.PREC == "amp" else None
        
        device_count = torch.cuda.device_count()
        if device_count > 1:
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.LOCOOP.PREC

        if prec == "amp":
            with autocast():
                output, output_local, img_feat_tea, img_feat_stu, text_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea = self.model(image, labels=label)
                all_text_features_tea = self.model.all_text_features_tea.clone()
                
                # Anti-NaN: Safe casting to float32 for CrossEntropy
                loss_id = F.cross_entropy(output.float(), label)
                loss_id2 = F.cross_entropy(output_local.float(), label)
                
                # GLAli Core: Distillation
                loss_distil_img = F.l1_loss(img_feat_tea.float(), img_feat_stu.float(), reduction='mean') * 10.0
                loss_distil_text = F.l1_loss(all_text_features_tea.float(), text_stu.float(), reduction='mean') * 25.0
                
                # GLAli Core: SupConLoss
                loss_supc = get_supc_loss(img_feat_stu.float(), id_loc_feats.float(), ood_loc_feats.float(), l2p, l2p_tea, label, n_class=self.model.n_cls, topk=self.top_k) * 0.5
                
                loss = loss_id + loss_id2 + loss_distil_img + loss_distil_text + loss_supc

            # Safety trigger to prevent NaN cascade
            if not torch.isfinite(loss):
                for name in self._optims:
                    if self._optims[name] is not None:
                        self._optims[name].zero_grad()
                return {"loss": 0.0, "acc": 0.0}

            for name in self._optims:
                if self._optims[name] is not None:
                    self._optims[name].zero_grad()
                    
    @torch.no_grad()
    def test_visualize(self, img_path, label_idx):
        """
        Generates a 14x14 heatmap showing which parts of the image 
        the model used to predict the given disease label.
        """
        self.set_model_mode("eval")
        
        # 1. Use the proper test transforms
        from dassl.data.transforms import build_transform
        from PIL import Image
        
        tfm_test = build_transform(self.cfg, is_train=False)
        image = Image.open(img_path).convert("RGB")
        image_tensor = tfm_test(image).unsqueeze(0).to(self.device)
        
        # 2. Extract image patches directly from the vision encoder
        _, local_features, _ = self.model.image_encoder(image_tensor.type(self.model.dtype))
        local_features = local_features / local_features.norm(dim=-1, keepdim=True) # Shape: [1, 196, 512]
        
        # 3. Get the text prototype for the requested class
        # text_features_tea contains the mean text embeddings for all classes [num_classes, 512]
        target_text = self.model.text_features_tea[label_idx] 
        
        # 4. Compute cosine similarity between the 196 patches and the text
        patch_scores = (local_features[0] @ target_text).float() # Shape: [196]
        
        # 5. Min-Max scale the scores so they look good on a heatmap (0 to 1)
        patch_scores = patch_scores - patch_scores.min()
        patch_scores = patch_scores / (patch_scores.max() + 1e-8)
        
        # 6. Reshape to a 14x14 grid
        heatmap = patch_scores.view(14, 14).cpu().numpy()
        
        return heatmap, image