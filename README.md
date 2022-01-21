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
