# Learning Scene Graph from General to Specific

![LICENSE](https://img.shields.io/badge/license-MIT-green)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

This anonymous repository contains code for the paper "Learning Scene Graph from General to Specific". This code is based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). Follow the [instructions](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) to install and use the code. 

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
    MODEL.PRETRAINED_MODEL_CKPT 'Path to the general model' \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  


