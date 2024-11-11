# ECCV2024-AdpatICMH
[ECCV2024] Image Compression for Machine and Human Vision With Spatial-Frequency Adaptation

[paper link](https://arxiv.org/abs/2407.09853).

## Absctract 
Image compression for machine and human vision (ICMH) has gained increasing attention in recent years. Existing ICMH methods are limited by high training and storage overheads due to heavy design of task-specific networks. To address this issue, in this paper, we develop a novel lightweight adapter-based tuning framework for ICMH, named Adapt-ICMH, that better balances task performance and bitrates with reduced overheads. We propose a spatial-frequency modulation adapter (SFMA) that simultaneously eliminates non-semantic redundancy with a spatial modulation adapter, and enhances task-relevant frequency components and suppresses task-irrelevant frequency components with a frequency modulation adapter. 
The proposed adapter is plug-and-play and compatible with almost all existing learned image compression models without compromising the performance of pre-trained models. Experiments demonstrate that Adapt-ICMH consistently outperforms existing ICMH frameworks on various machine vision tasks with fewer fine-tuned parameters and reduced computational complexity.

## Install

```bash
git clone https://github.com/qingshi9974/ECCV2024-AdpatICMH
pip install compressai
pip install timm tqdm click
```

Install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for object detection and instance segementation.

## Dataset
The following datasets are used and needed to be downloaded.
- ImageNet1K
- COCO 2017 Train/Val
- Kodak

## Example Usage
Specify the data paths, target rate point, corresponding lambda, and checkpoint in the config file accordingly.


### Classification
`python examples/classification.py -c config/classification.yaml`<br>
Add argument `-T` for evaluation.

### Object Detection
`python examples/detection.py -c config/detection.yaml`<br>
Add argument `-T` for evaluation.

### Instance Segmentation
`python examples/segmentation.py -c config/segmentation.yaml`<br>
Add argument `-T` for evaluation.

## Pre-trained Weights for TIC
|         Tasks         |       |       |       |       |
|:---------------------:|-------|-------|-------|-------|
|     Base codec (TIC)  | [1](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/base_codec_1.pth.tar) | [2](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/base_codec_2.pth.tar) | [3](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/base_codec_3.pth.tar) | [4](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/base_codec_4.pth.tar) |


## Citation
If you find our project useful, please cite the following paper.
```

@inproceedings{li2024image,
  title={Image compression for machine and human vision with spatial-frequency adaptation},
  author={Li, Han and Li, Shaohui and Ding, Shuangrui and Dai, Wenrui and Cao, Maida and Li, Chenglin and Zou, Junni and Xiong, Hongkai},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```

## Ackownledgement
Our work is based on the framework of [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [TransTIC](https://github.com/NYCU-MAPL/TransTIC). 


