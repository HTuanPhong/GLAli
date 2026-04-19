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


def get_dense_logits2(image_features, local_image_features, all_text_features, mean_text_features, topk=50):
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
        self.cfg = cfg

        description_file = os.path.join('./description', f'{cfg.DATASET.NAME}.json')
        print(f'Using description file: {description_file}')
        llm_descriptions = json.load(open(description_file))
        
        text_features = []
        template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        all_prompt =[]
        
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
        
        self.all_text_features_tea = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_mean = text_features.mean(dim=0)
        self.text_features_tea = text_features_mean / text_features_mean.norm(dim=-1, keepdim=True)
        self.text_prototypes = self.all_text_features_tea   

        if cfg.is_bonder:
            self.bonder = CrossAttnBlock(512)
            self.bonder.to(self.dtype)

    def forward(self, image, mask=None, labels=None):
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

            sim = local_image_features @ (l2p.mean(dim=1, keepdim=True).transpose(1, 2))
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
            
            contra_labels = torch.arange(c).view(-1, 1).cuda()
            mask_labels = torch.eq(labels.unsqueeze(1), contra_labels.T).to(self.dtype).cuda()
            
            update_features = torch.matmul(mask_labels.view(bs, c).transpose(0, 1).unsqueeze(0).repeat(n_disc-1, 1, 1), text_bias.transpose(1, 0))
            
            proto_mask = torch.zeros(c, dtype=torch.int).cuda()
            proto_mask[labels] = 1
            proto_mask = proto_mask.view(1, -1, 1).repeat(n_disc, 1, d)
            
            update_features = torch.cat([self.text_prototypes[0:1, :, :], update_features], dim=0)
            updated_proto = (1 - proto_mask) * updated_proto + proto_mask * (alpha * updated_proto + (1 - alpha) * update_features)

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

        if cfg.TRAINER.LOCOOP.PREC in["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Applying Pure GLAli + ProLIP: Unfreezing Attention, Bonder, and Visual Projection")
        
        for name, param in self.model.named_parameters():
            # EXACT Integration: GLAli's Attention & Bonder + ProLIP's Projection Matrix
            if 'image_encoder.transformer.resblocks.11.attn' in name or 'bonder' in name or 'image_encoder.proj' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        self.model.to(self.device)
        
        # Save a frozen copy of the initial projection matrix for ProLIP MSE Regularization
        self.proj_init = self.model.image_encoder.proj.detach().clone().to(self.device)

        # Single Unified Optimizer
        trainable_params =[p for p in self.model.parameters() if p.requires_grad]
        self.optim = build_optimizer(trainable_params, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.LOCOOP.PREC == "amp" else None

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
                
                # --- THE FIX: We deleted loss_distil_img. ProLIP handles vision safety now! ---
                loss_distil_text = F.l1_loss(all_text_features_tea, text_stu, reduction='mean') * 25
                
                # GLAli LocSC Loss
                loss_supc = get_supc_loss(img_feat_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea, label, topk=self.top_k) * 0.5
                
                # ProLIP Weight Regularization (Replaces Image Distillation)
                loss_prolip = torch.sum((self.model.image_encoder.proj - self.proj_init) ** 2)
                lambda_prolip = 1.0 / self.cfg.DATASET.NUM_SHOTS 
                
                # Perfect Synergy Loss
                loss = loss_id + loss_id2 + loss_distil_text + loss_supc + (lambda_prolip * loss_prolip)

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, output_local, img_feat_tea, img_feat_stu, text_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea = self.model(image, labels=label)
            all_text_features_tea = self.model.all_text_features_tea.clone()
            
            loss_id = F.cross_entropy(output, label)
            loss_id2 = F.cross_entropy(output_local, label)
            
            # --- THE FIX: We deleted loss_distil_img. ProLIP handles vision safety now! ---
            loss_distil_text = F.l1_loss(all_text_features_tea, text_stu, reduction='mean') * 25
            loss_supc = get_supc_loss(img_feat_stu, id_loc_feats, ood_loc_feats, l2p, l2p_tea, label, topk=self.top_k) * 0.5
            
            # ProLIP Weight Regularization
            loss_prolip = torch.sum((self.model.image_encoder.proj - self.proj_init) ** 2)
            lambda_prolip = 1.0 / self.cfg.DATASET.NUM_SHOTS 
            
            # Perfect Synergy Loss
            loss = loss_id + loss_id2 + loss_distil_text + loss_supc + (lambda_prolip * loss_prolip)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        loss_summary = {
            "loss": loss.item(),
            "loss_id": loss_id.item(),
            "loss_prolip": loss_prolip.item(),
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
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]

            print(f'Loading weights to {name} from "{model_path}"')
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
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
            self.model.text_prototypes = torch.load(osp.join(self.output_dir, 'proto.pth'))
            
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            
            if len(output) >= 2:
                if self.cfg.is_bonder:
                    output = output[1] + 0.05 * output[0]
                else:
                    output = output[0]
                    
            # DELETED: self.label.append(label) - This was causing the crash!
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def test_ood(self, data_loader, T):
        self.model.image_features_store =[]
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.evaluator.reset()

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

    def model_inference(self, input):
        return self.model(input)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

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