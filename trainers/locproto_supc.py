import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY # , TrainerX
from utils.trainer import TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip_w_local import clip_clear as clip
from clip_w_local.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
from tqdm import tqdm
from PIL import Image
from .zsclip_contra import entropy_select_topk2, CUSTOM_TEMPLATES
import os
import json
from copy import deepcopy
from timm.models.layers import DropPath, Mlp
from utils.bonder import CrossAttnBlock
from utils.loss import SupConLoss

# Imports needed to build pristine cache for Tip-Adapter
from utils.data_manager import build_data_loader
from dassl.data.transforms import build_transform

_tokenizer = _Tokenizer()
softmax = nn.Softmax(dim=1).cuda()


def entropy_select_topk(p, top_k, label, num_of_local_feature):
    """
    Extract non-Top-K regions and calculate entropy.
    """
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
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def get_dense_logits2(image_features, local_image_features, all_text_features, mean_text_features, topk=50):
    base_logits = image_features @ mean_text_features.T   #[bs, 512] *[N, 512] -> [bs, N]
    image_features = image_features.unsqueeze(1)  #[bs, 1, 512]
    all_image_features = local_image_features
    w = torch.einsum('bmd,bnd->bmn', image_features, all_image_features) #[bs, 1, 197]

    mean_text_features = mean_text_features.unsqueeze(0) #[n_desc, N, 512]
    _,n_cls,d = mean_text_features.shape
    all_text_features = all_text_features.reshape(-1, n_cls, d)
    v = torch.einsum('mcd,ncd->mnc', mean_text_features, all_text_features)  #  [1, N, 512] *[n_desc, N, 512] -> [1, n_desc, N]
    v = F.softmax(v, dim=1)
    sim = torch.einsum('bmd,ncd->bcmn', all_image_features, all_text_features)  #[bs, 197, 512] *[n_desc, N, 512] ->[bs, N, 197, n_desc]
    sim, idx = sim.topk(dim=2, k=topk)    # [bs, N, k, n_desc]
    idx = idx[:, 0, :, 0].unsqueeze(1)
    w = torch.gather(w, dim=2, index=idx)
    w = F.softmax(w, dim=-1)
    weight = torch.einsum('bdm,dnc->bcmn', w,v) #[bs, N, 197, n_desc]
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
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, cache_keys=None, cache_values=None):
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

            # get descriptions
            for i in range(50):
                prompt_desc = prompt + ' ' + llm_descriptions[classname.replace("_", " ")][i]
                prompts.append(prompt_desc)
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
            all_prompt.append(prompts)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    text_features.append(clip_model.encode_text(prompts)) # n_desc x d
                    
        self.all_prompt = torch.cat(all_prompt)

        text_features = torch.cat(text_features) # (n_cls x n_desc) x d
        _, d = text_features.shape
        self.ndisc = 51
        text_features = text_features.view(self.ndisc, -1, d)
        self.all_text_features_tea = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_mean = text_features.mean(dim=0)
        self.text_features_tea = text_features_mean / text_features_mean.norm(dim=-1, keepdim=True)
        self.text_prototypes = self.all_text_features_tea    # ndisc, C, d

        # Bonder
        if cfg.is_bonder:
            self.bonder = CrossAttnBlock(512)
            self.bonder.to(self.dtype)

        # ---------------- TIP-ADAPTER-F MEMORY CACHE ----------------
        self.tip_adapter = None
        if cache_keys is not None:
            print("Initializing Tip-Adapter-F Cache Parameters...")
            self.tip_alpha = nn.Parameter(torch.tensor(1.0, dtype=self.dtype))
            self.tip_beta = 5.5   
            # Enable the cached keys to be learnable
            self.tip_adapter = nn.Linear(cache_keys.shape[1], cache_keys.shape[0], bias=False).to(self.dtype).cuda()
            self.tip_adapter.weight = nn.Parameter(cache_keys) 
            self.register_buffer("cache_values", cache_values.to(self.dtype).cuda())

    def forward(self, image, mask=None, labels = None):
        updated_proto = None
        # teacher model
        with torch.no_grad():
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

        # ---------------- TIP-ADAPTER-F LOGIC ----------------
        if self.tip_adapter is not None:
            affinity = self.tip_adapter(image_features)
            cache_logits = torch.exp(-self.tip_beta * (1.0 - affinity)) @ self.cache_values
            
            # Tip-Adapter's output scaled properly by logit_scale so it competes cleanly with GLAli
            scaled_cache_logits = cache_logits * self.tip_alpha * logit_scale
            
            logits = logits + scaled_cache_logits
            logits_local = logits_local + scaled_cache_logits
        # -----------------------------------------------------

        return logits, logits_local, image_features_tea, image_features, updated_proto_norm, id_loc_feats, ood_loc_feats, l2p, l2p_tea


@TRAINER_REGISTRY.register()
class LocProto(TrainerX):
    """Local regularized Context Optimization (LoCoOp).
    """

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

        if cfg.TRAINER.LOCOOP.PREC in ["fp32", "amp"]:
            # CLIP's default precision is fp16
            clip_model.float()

        # ---------------- CONSTRUCT PRISTINE TIP-ADAPTER CACHE ----------------
        print("Extracting Pristine Visual Memory Cache from training set (Tip-Adapter)...")
        # We must use unaugmented data to build the cache!
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
            for batch in tqdm(cache_loader, desc="Building Cache"):
                image = batch["img"].to(self.device)
                label = batch["label"].to(self.device)
                
                img_feat, _, _ = clip_model.visual(image.type(clip_model.dtype))
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                
                cache_keys.append(img_feat.cpu())
                cache_labels.append(label.cpu())
                
        cache_keys = torch.cat(cache_keys, dim=0).to(self.device) 
        cache_labels = torch.cat(cache_labels, dim=0).to(self.device) 
        cache_values = F.one_hot(cache_labels, num_classes=len(classnames)).float().to(self.device) 
        # ----------------------------------------------------------------------

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, cache_keys=cache_keys, cache_values=cache_values)

        print("Configuring Gradients: Vision Encoder + Bonder + Tip-Adapter Keys")
        for name, param in self.model.named_parameters():
            if 'image_encoder.transformer.resblocks.11.attn' in name or 'bonder' in name or 'tip_adapter' in name or 'tip_alpha' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        self.model.to(self.device)
        
        # NOTE: only give prompt_learner to the optimizer
        if "ViT" in cfg.MODEL.BACKBONE.NAME:
            self.optim = build_optimizer(self.model.image_encoder.transformer.resblocks[-1].attn, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("attn_learner", self.model.image_encoder.transformer.resblocks[-1].attn, self.optim,
                                self.sched)
            
            if cfg.is_bonder:
                cfg.OPTIM2 = deepcopy(cfg.OPTIM)
                cfg.OPTIM2.LR = cfg.OPTIM.LR
                self.optim2 = build_optimizer(self.model.bonder, cfg.OPTIM2)
                self.sched2 = build_lr_scheduler(self.optim2, cfg.OPTIM2)
                self.register_model("bonder_learner", self.model.bonder, self.optim2,
                                    self.sched2)

            # ---------------- REGISTER TIP-ADAPTER OPTIMIZER ----------------
            if hasattr(self.model, "tip_adapter") and self.model.tip_adapter is not None:
                cfg.OPTIM_TIP = deepcopy(cfg.OPTIM)
                # Tip-Adapter Learns well with a higher learning rate (e.g. 0.001)
                cfg.OPTIM_TIP.LR = 0.001
                tip_params =[self.model.tip_adapter.weight, self.model.tip_alpha]
                self.optim_tip = build_optimizer(tip_params, cfg.OPTIM_TIP)
                self.sched_tip = build_lr_scheduler(self.optim_tip, cfg.OPTIM_TIP)
                self.register_model("tip_adapter_learner", self.model.tip_adapter, self.optim_tip, self.sched_tip)
            # ----------------------------------------------------------------

        elif "RN" in cfg.MODEL.BACKBONE.NAME:
            self.optim = build_optimizer(self.model.image_encoder.attnpool, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("attn_learner", self.model.image_encoder.attnpool, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.LOCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

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

            # CRITICAL FIX: Ensure all registered optimizers (including Tip-Adapter) are stepped in AMP mode!
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
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.model.image_features_store =[]
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "test":
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader
        else:
            split = "train"
            data_loader = self.train_loader_x

        print(f"Evaluate on the *{split}* set")

        if self.cfg.is_bonder:
            self.model.text_prototypes = torch.load(osp.join(self.output_dir, 'proto.pth'))
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
        """Test-time OOD detection pipeline."""
        self.model.image_features_store =[]
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.evaluator.reset()

        glmcm_score =[]
        mcm_score =[]
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            (images, labels, *id_flag) = batch
            if isinstance(images, str):
                images, label = self.parse_batch_test(batch)
            else:
                images = images.cuda()
            images = images.cuda()
            output, output_local, _, _, _, _, _, _, _ = self.model_inference(images)
            if self.cfg.is_bonder:
                output = output_local + 0.05 * output
            output /= 100.0
            output_local /= 100.0
            smax_global = to_np(F.softmax(output/T, dim=-1))  
            smax_local = to_np(F.softmax(output_local/T, dim=-1))
            mcm_global_score = -np.max(smax_global, axis=1)
            mcm_score.append(mcm_global_score)

        return concat(mcm_score)[:len(data_loader.dataset)].copy(), concat(mcm_score)[:len(data_loader.dataset)].copy(), concat(mcm_score)[:len(data_loader.dataset)].copy(), concat(mcm_score)[:len(data_loader.dataset)].copy()

    @torch.no_grad()
    def test_ood1(self, data_loader, T):
        """Test-time OOD detection pipeline."""
        # self.model.image_features_store =[]
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.evaluator.reset()

        glmcm_score =[]
        mcm_score = []
        loc_score =[]
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            (images, labels, *id_flag) = batch
            if isinstance(images, str):
                images, label = self.parse_batch_test(batch)
                labels = label
            else:
                images = images.cuda()
                labels = labels.cuda()
            images = images.cuda()
            output, output_local, zs_output, zs_local = self.model_inference(images)
            pred = torch.argmax(output, dim=-1)
            batch_size, num_of_local_feature, _ = output_local.shape
            output_local_ = zs_local.view(batch_size * num_of_local_feature, -1)

            selected = entropy_select_topk2(p=output_local_, top_k=self.cfg.topk, label=labels)
            selected = selected.view(batch_size, num_of_local_feature)
            attention_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=output.device)
            attention_mask = torch.cat((attention_mask, selected), dim=1)
            output2, _, zs_output2, _ = self.model_inference(images, mask=attention_mask)

            output /= 100.0
            output_local /= 100.0
            output2 /= 100.0
            smax_global0 = to_np(F.softmax(output/T, dim=-1))
            smax_global = to_np(output)
            smax_local = to_np(F.softmax(output_local/T, dim=-1))
            smax_global2 = to_np(output2)
            mcm_global_score = -np.max(smax_global, axis=1)
            mcm_global_score0 = -np.max(smax_global0, axis=1)
            contr_score = -np.max(np.abs(smax_global+0.5*(to_np(zs_output)-to_np(zs_output2))), axis=1)
            mcm_local_score = -np.max(smax_local, axis=(1, 2))
            mcm_score.append(mcm_global_score)
            glmcm_score.append(contr_score)
            loc_score.append(mcm_local_score)

        return concat(mcm_score)[:len(data_loader.dataset)].copy(), concat(glmcm_score)[:len(data_loader.dataset)].copy(), concat(loc_score)[:len(data_loader.dataset)].copy()

    @torch.no_grad()
    def test_visualize(self, img_path, label):
        """code for visualization results"""
        self.set_model_mode("eval")
        self.evaluator.reset()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/16", device=device)

        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        output, output_local = self.model_inference(image)

        num_regions = output_local.shape[1]
        label = torch.tensor(label).cuda()
        label_repeat = label.repeat_interleave(num_regions)
        output_local = F.softmax(output_local, dim=-1)

        output_local = output_local.view(num_regions, -1)

        # -----top 200--------
        pred_topk = torch.topk(output_local, k=200, dim=1)[1]
        contains_label = pred_topk.eq(torch.tensor(label_repeat).unsqueeze(1)).any(dim=1)

        return contains_label