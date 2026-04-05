import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY # , TrainerX
from utils.trainer import TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip_w_local import clip_clear as clip
# from clip_w_local import clip as clip0
from clip_w_local.model import convert_weights

from .zsclip_clear import load_clip_to_cpu
# from .locoop import load_clip_to_cpu as load_clip_to_cpu0
# from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
import numpy as np
from tqdm import tqdm
import os
import json

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}.", # , a type of food
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "Skin40": "a photo of a {}, a type of skin disease.",
    "chest": "a photo of a {}.",
    "ISIC": "a photo of a {}.",   # , a type of skin disease
    "Dermnet": "a photo of a {}, a type of skin disease.",
    "RFMiD": "a photo of a {}, a type of fundus disease.",
    "ISIC2": "a photo of a {}, a type of skin disease.",
}

def entropy_select_topk(p, top_k, label, num_of_local_feature=196):
    """
    Extract non-Top-K regions and calculate entropy.
    """
    label_repeat = label.repeat_interleave(num_of_local_feature, dim=0)   # [100, 2]
    p = F.softmax(p, dim=-1)
    pred_topk = torch.topk(p, k=top_k, dim=-1)[1]
    contains_label = pred_topk.eq(label_repeat.unsqueeze(1)).any(dim=1)
    # contains_label = (pred_topk.unsqueeze(1) == label_repeat.unsqueeze(2)).any(dim=(1,2))    # topk对topk
    # contains_label = (pred_topk.view(label.shape[0], num_of_local_feature, 1, 1) == label_repeat.view(label.shape[0], num_of_local_feature, 1, 1)).any(dim=(1,2))      # topk个label对1个patch预测
    num_mask=contains_label.view(label.shape[0],196).sum(dim=1)
    # selected_p = p[~contains_label]

    # if selected_p.shape[0] == 0:
    #     return torch.tensor([0]).cuda()
    # return -torch.mean(torch.sum(selected_p * torch.log(selected_p+1e-5), 1))
    return contains_label

def entropy_select_topk2(p, top_k, label, num_of_local_feature=196):
    """
    Extract non-Top-K regions and calculate entropy.
    """
    bs = label.shape[0]
    label_repeat = label.repeat_interleave(num_of_local_feature)   # [6272, 1000]
    p = F.softmax(p, dim=-1)
    entro = torch.sum(p * torch.log(p+1e-5), 1)
    entro = entro.view(bs, 196)
    topk_idx = torch.topk(entro, k=top_k, dim=1)[1]   # 熵最小的k个patch
    mask = torch.zeros((bs, num_of_local_feature), dtype=torch.bool, device=p.device)
    mask.scatter_(1, topk_idx, True)

    return mask

def get_avg_entro(p):
    p = F.softmax(p, dim=-1)
    entro = torch.sum(p * torch.log(p+1e-5), -1)
    # entro = entro.view(p.shape[0], 196)
    avg_entro = entro.mean(dim=-1)
    return avg_entro

@TRAINER_REGISTRY.register()
class ZeroshotCLIP_Contra(TrainerX):
    def build_model(self):
        cfg = self.cfg
        if "mnist" in cfg.in_dataset:
            classnames = cfg.classnames
        else:
            classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        # clip_ori_model = load_clip_to_cpu0(cfg)
        clip_model.to(self.device)
        # clip_ori_model.to(self.device)

        multi_description = False
        if multi_description:
            description_file = os.path.join('./description', f'{cfg.DATASET.NAME}.json')
            print(f'Using description file: {description_file}')
            llm_descriptions = json.load(open(description_file))
            text_features = []
            template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            for classname in classnames:
                prompts = []
                prompt = template.format(classname.replace("_", " "))
                prompts.append(prompt + '.')

                # get descriptions
                # assert len(llm_descriptions[classname]) >= args.num_descriptor
                for i in range(50):
                    prompt_desc = prompt + '. ' + llm_descriptions[classname][i]   # n条description都和 a photo of  a ... 拼接然后单独处理
                    prompts.append(prompt_desc)
                prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()

                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        text_features.append(clip_model.encode_text(prompts)) # n_desc x d

            text_features = torch.cat(text_features) # (n_cls x n_desc) x d
            _, d = text_features.shape
            ndisc = 50
            # 求平均作为用来预测的text embedding
            text_features = text_features.view(ndisc, -1, d)
            text_features = text_features.mean(dim=0)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            print(f"Prompts: {prompts}")
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            prompts = prompts.to(self.device)

            with torch.no_grad():
                text_features = clip_model.encode_text(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model
        # self.clip_ori_model = clip_ori_model

    def model_inference(self, image, mask=None):
        image_features, local_feats, _ = self.clip_model.encode_image(image, mask=mask)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_features = local_feats / local_feats.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        local_logits = logit_scale * local_features @ self.text_features.t()
        return logits, local_logits
    
    def ori_model_inference(self, image, mask=None):
        image_features, local_feats, _ = self.clip_ori_model.encode_image(image, mask=mask)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_features = local_feats / local_feats.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_ori_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        local_logits = logit_scale * local_features @ self.text_features.t()
        return logits, local_logits

    def model_forward(self, image, mask=None):
        image_features, local_feats, _ = self.clip_model.encode_image(image, mask=mask)
        ori_image_features, ori_local_feats, _ = self.clip_ori_model.encode_image(image, mask=mask)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_feats = local_feats / local_feats.norm(dim=-1, keepdim=True)
        ori_image_features = ori_image_features / ori_image_features.norm(dim=-1, keepdim=True)
        ori_local_feats = ori_local_feats / ori_local_feats.norm(dim=-1, keepdim=True)
        return image_features, local_feats, ori_image_features, ori_local_feats
    
    @torch.no_grad()
    def test_ood(self, data_loader, T):
        """Test-time OOD detection pipeline."""
        # self.model.image_features_store = []
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.evaluator.reset()

        glmcm_score = []
        mcm_score = []
        loc_score = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            (images, labels, *id_flag) = batch
            if isinstance(images, str):
                images, label = self.parse_batch_test(batch)
                labels = label
            else:
                images = images.cuda()
                labels = labels.cuda()
            images = images.cuda()
            output, output_local = self.model_inference(images)
            # pred_topk = torch.topk(output, k=self.cfg.topk, dim=-1)[1]
            # top2_correct = (pred_topk == labels.view(-1, 1)).any(dim=1).float().mean()
            # pred = torch.argmax(output, dim=-1)
            batch_size, num_of_local_feature, _ = output_local.shape
            output_local_ = output_local.view(batch_size * num_of_local_feature, -1)

            selected = entropy_select_topk2(p=output_local_, top_k=self.cfg.topk, label=labels)  # 取正确结果在top-k里面的出来,mask掉
            selected = selected.view(batch_size, num_of_local_feature)
            # attention_mask = torch.zeros((batch_size, num_of_local_feature+1))
            # attention_mask[selected] = 1
            attention_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=output.device)
            attention_mask = torch.cat((attention_mask, selected), dim=1)
            output2, _ = self.model_inference(images, mask=attention_mask)

            output /= 100.0
            output_local /= 100.0
            output2 /= 100.0
            contra_logits = output + 0.2*torch.abs((output-output2))
            smax_contra = to_np(F.softmax(contra_logits/T, dim=-1))
            smax_global0 = to_np(F.softmax(output/T, dim=-1))
            smax_global = to_np(output/T)
            smax_local = to_np(F.softmax(output_local/T, dim=-1))
            # smax_global2 = to_np(F.softmax(output2/T, dim=-1))
            smax_global2 = to_np(output2/T)
            mcm_global_score = -np.max(smax_global, axis=1)
            mcm_global_score0 = -np.max(smax_global0, axis=1)
            contr_score = -np.max(np.abs(smax_global+0.5*(smax_global-smax_global2)), axis=1)  # smax_global+0.5*
            # contr_score = -np.max(smax_contra, axis=1)
            # mcm_global_score = -(np.abs(to_np(torch.gather(smax_global, 1, pred.unsqueeze(-1))-torch.gather(smax_global2, 1, pred.unsqueeze(-1))).squeeze(-1)))
            # mcm_global_score = -to_np(1.5*smax_global[pred] - 0.5*smax_global2[pred])
            # mcm_global_score = -to_np((1.5*torch.gather(smax_global, 1, pred.unsqueeze(-1)) - 0.5*torch.gather(smax_global2, 1, pred.unsqueeze(-1))).squeeze(-1))
            mcm_local_score = -np.max(smax_local, axis=(1, 2))
            mcm_score.append(mcm_global_score0)
            glmcm_score.append(contr_score)   # mcm_global_score+mcm_local_score
            loc_score.append(mcm_global_score0)

        return concat(mcm_score)[:len(data_loader.dataset)].copy(), concat(glmcm_score)[:len(data_loader.dataset)].copy(), concat(loc_score)[:len(data_loader.dataset)].copy(), concat(loc_score)[:len(data_loader.dataset)].copy()
    
    @torch.no_grad()
    def get_vir(self, data_loader, T):
        """Test-time OOD detection pipeline."""
        # self.model.image_features_store = []
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.evaluator.reset()

        sim = []
        ori_sim = []
        entro_lst = []
        ori_entro_lst = []
        loc_score = []
        loc_ori_score = []
        feats = []
        ori_feats = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            (images, labels, *id_flag) = batch
            if isinstance(images, str):
                images, label = self.parse_batch_test(batch)
                labels = label
            else:
                images = images.cuda()
                labels = labels.cuda()
            images = images.cuda()
            # image_features, local_feats, ori_image_features, ori_local_feats = self.model_forward(images)
            # feats.append(to_np(local_feats.view(-1, 512)))
            # ori_feats.append(to_np(ori_local_feats.view(-1, 512)))
            # break
            _, loc_output = self.model_inference(images)
            _, loc_output_ori = self.ori_model_inference(images)
            # 计算平均entropy
            clear_entro = get_avg_entro(loc_output)
            ori_entro = get_avg_entro(loc_output_ori)
            entro_lst.append(to_np(clear_entro))
            ori_entro_lst.append(to_np(ori_entro))
            # 计算平均和cls token的相似性
            # cls_loc_sim = image_features.unsqueeze(1) @ local_feats.permute(0, 2, 1)
            # cls_loc_sim = cls_loc_sim.mean(dim=-1).squeeze(-1)
            # sim.append(to_np(cls_loc_sim))
            # cls_loc_sim_ori = ori_image_features.unsqueeze(1) @ ori_local_feats.permute(0, 2, 1)
            # cls_loc_sim_ori = cls_loc_sim_ori.mean(dim=-1).squeeze(-1)
            # ori_sim.append(to_np(cls_loc_sim_ori))
            # 计算loc的ood得分
            # loc_output /= 100.0
            # loc_output_ori /= 100.0
            # smax_local = to_np(F.softmax(loc_output/T, dim=-1))
            # smax_local_ori = to_np(F.softmax(loc_output_ori/T, dim=-1))
            # mcm_local_score = -np.max(smax_local, axis=(1, 2))
            # mcm_local_ori_score = -np.max(smax_local_ori, axis=(1, 2))
            # loc_score.append(mcm_local_score)
            # loc_ori_score.append(mcm_local_ori_score)

        # return to_np(clear_entro), to_np(ori_entro), to_np(cls_loc_sim), to_np(cls_loc_sim_ori), concat(loc_score)[:len(data_loader.dataset)].copy(), concat(loc_ori_score)[:len(data_loader.dataset)].copy()
        return concat(entro_lst).copy(), concat(ori_entro_lst).copy()