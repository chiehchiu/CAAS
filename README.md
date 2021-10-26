
## Introduction

Modified from mmclassification. 
Support 3D resnet pretraining(see `configs/imagenet_pretrain/`).
Support LIDC-IDRI dataset(see `configs/lidc/`).

## Installation

Please refer to [install.md](scripts/install.md) for installation and dataset preparation.


## Training

```
bash scripts/train.sh train
```

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:
- [x] ResNet
- [x] ResNeXt
- [x] SE-ResNet
- [x] SE-ResNeXt
- [x] RegNet
- [x] ShuffleNetV1
- [x] ShuffleNetV2
- [x] MobileNetV2
- [x] MobileNetV3


