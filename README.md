# Synthesis in Style

This repository contains the code for the paper "Synthesis in Style: Semantic Segmentation of Historical Documents using Synthetic Data".
The trained models and benchmark dataset used for evaluation can be found [here](https://bartzi.de/research/synthesis_in_style).

## Installation

In `stylegan_code_finder` directory execute: 
```
pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

In the root directory execute:
``` 
git submodule init
git submodule update
cd training_tools && pip3 install -e .
```

## Usage

For information how to use the code in this repository, please refer to the [wiki](https://github.com/hendraet/synthesis-in-style/wiki/How-to-Train-a-Segmentation-Model)

## License and Credits

The code used for the [EMANet](https://arxiv.org/abs/1907.13426) implementation is taken from [here](https://github.com/XiaLiPKU/EMANet).

The code for [TransUNet](https://arxiv.org/abs/2102.04306) was taken from [this](https://github.com/Beckschen/TransUNet) repository.

StyleGAN2 implementation and custom CUDA kernel codes are taken from official the official [NVIDIA repository](https://github.com/NVlabs/stylegan2), which is published under Nvidia Source Code License-NC.

