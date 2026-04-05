import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY # , TrainerX
from utils.trainer import TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip_w_local import clip_clear as clip
from clip_w_local.model import convert_weights

# from .coop import load_clip_to_cpu
# from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
import numpy as np
from tqdm import tqdm

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "Skin40": "a photo of a {}",
    "chest": "a photo of a {}",
}

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

@TRAINER_REGISTRY.register()
class ZeroshotCLIP_Clear(TrainerX):
    def build_model(self):
        cfg = self.cfg
        if "mnist" in cfg.in_dataset:
            classnames = cfg.classnames
        else:
            classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

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

    def model_inference(self, image):
        image_features, local_feats, _ = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_features = local_feats / local_feats.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        local_logits = logit_scale * local_features @ self.text_features.t()
        return logits, local_logits
    
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
        locmcm_score = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            (images, labels, *id_flag) = batch
            if isinstance(images, str):
                images, label = self.parse_batch_test(batch)
            else:
                images = images.cuda()
            images = images.cuda()
            output, output_local = self.model_inference(images)
            output /= 100.0
            output_local /= 100.0
            smax_global = to_np(F.softmax(output/T, dim=-1))
            smax_local = to_np(F.softmax(output_local/T, dim=-1))
            mcm_global_score = -np.max(smax_global, axis=1)
            mcm_local_score = -np.max(smax_local, axis=(1, 2))
            mcm_score.append(mcm_global_score)
            glmcm_score.append(mcm_global_score+mcm_local_score)  # mcm_global_score+mcm_local_score
            locmcm_score.append(mcm_local_score)

        return concat(mcm_score)[:len(data_loader.dataset)].copy(), concat(glmcm_score)[:len(data_loader.dataset)].copy(), concat(locmcm_score)[:len(data_loader.dataset)].copy()



# @TRAINER_REGISTRY.register()
# class ZeroshotCLIP2(ZeroshotCLIP):
#     """Prompt ensembling."""

#     # templates = IMAGENET_TEMPLATES
#     templates = IMAGENET_TEMPLATES_SELECT

#     def build_model(self):
#         cfg = self.cfg
#         classnames = self.dm.dataset.classnames

#         print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
#         clip_model = load_clip_to_cpu(cfg)
#         clip_model.to(self.device)

#         for params in clip_model.parameters():
#             params.requires_grad_(False)

#         # add custom-made prompt
#         if cfg.DATASET.NAME != "ImageNet":
#             self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

#         num_temp = len(self.templates)
#         print(f"Prompt ensembling (n={num_temp})")

#         mean_text_features = 0
#         for i, temp in enumerate(self.templates):
#             prompts = [temp.format(c.replace("_", " ")) for c in classnames]
#             prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
#             text_features = clip_model.encode_text(prompts)
#             text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#             mean_text_features = mean_text_features + text_features
#         mean_text_features = mean_text_features / num_temp
#         mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

#         self.text_features = mean_text_features
#         self.clip_model = clip_model
