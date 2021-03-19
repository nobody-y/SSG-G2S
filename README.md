# From General to Specific: Informative Scene Graph Generation via Balance Adjustment

![LICENSE](https://img.shields.io/badge/license-MIT-green)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

This anonymous repository contains code for the paper "From General to Specific: Informative Scene Graph Generation via Balance Adjustment". This code is based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). Follow the [instructions](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) to install and use the code. In this work, balance adjustment scene graph generation (BA-SGG) is proposed for learning scene graphs from general to specific. We first reveal the imbalance between common predicates and informative predicates. Then two strategies are designed: Semantic Adjustment and Balanced Predicate Learning, in view of the imbalance. The semantic adjustment strategy explores the semantic relation between predicates, and the balanced predicate learning strategy transfers the model to informative predicates.
## Framework
<div align=center><img width="672" height="508" src=demo/framework_G2ST.png/></div>
Whole training process of our method, \ie, Semantic Adjustment (SA) and Balanced Predicate Learning (BPL).

## Performance
<div align=center><img width="938" height="345" src=demo/performance.png/></div>
Comparison between our method (BA-SGG) and previous methods.

## Visualization
<div align=center><img width="994" height="774" src=demo/vis_res_supp1.png/></div>
Visualization results of Transformer (BA-SGG) on the PredCls task. Our approach adapts to these complex scenes appropriately and generates some informative predicates, such as walking on in the first image, riding in the fifth image and looking at in the eighth image.

## Training and Testing 
We write some [scripts](https://github.com/nobody-y/SSG-G2S/tree/main/scripts) for training and testing.
The training process is divided into two stages:
### Training the general model
The training script should be set up as follows: \
    MODEL.PRETRAINED_MODEL_CKPT '' \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  
### Finetuning the specific model
The training script should be set up as follows: \
    MODEL.PRETRAINED_MODEL_CKPT 'path to the general model' \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  

Trained models can be dowload from [BaiduYun](https://pan.baidu.com/s/1s-jk8GsCAgCDv6XdZRA8jA) (Password: a4rd)

