# Global and Local Vision-Language Alignment for Few-Shot Learning and Few-Shot OOD Detection


## Requirement
### Package
Our experiments are conducted with Python 3.8 and Pytorch 1.8.1.

All required packages are based on [CoOp](https://github.com/KaiyangZhou/CoOp) (for training) and [MCM](https://github.com/deeplearning-wisc/MCM) (for evaluation).
This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `LoCoOp/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) and [MCM](https://github.com/deeplearning-wisc/MCM) (this should be done when `dassl` is activated).


## Quick Start
### 1. Training
The training script is in `./scripts/GLAli/train.sh`.

e.g., 16-shot training with ViT-B/16
```train
CUDA_VISIBLE_DEVICES=0 bash scripts/locoop/train.sh
```


### 2. Inference 
The inference script is in `./scripts/GLAli/eval.sh`.


## Acknowledgement
We adopt these codes to create this repository.
* [Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557), in CVPR, 2022.
* [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134), IJCV, 2022.
* [Delving into Out-of-Distribution Detection with Vision-Language Representations](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e43a33994a28f746dcfd53eb51ed3c2d-Abstract-Conference.html), in NeurIPS, 2022
* [Zero-Shot In-Distribution Detection in Multi-Object Settings Using Vision-Language Foundation Models](https://arxiv.org/abs/2304.04521), arXiv, 2023
* [LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/f0606b882692637835e8ac981089eccd-Paper-Conference.pdf), in NeurIPS, 2023


## Citaiton
If you find our work interesting or use our code/models, please consider citing:
```bibtex
@InProceedings{YanJie_Global_MICCAI2025,
        author = { Yan, Jie AND Guan, Xiaoyuan AND Zheng, Wei-Shi AND Chen, Hao AND Wang, Ruixuan},
        title = { { Global and Local Vision-Language Alignment for Few-Shot Learning and Few-Shot OOD Detection } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
        year = {2025},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15964},
        month = {September},
        page = {208 -- 218}
}
```
