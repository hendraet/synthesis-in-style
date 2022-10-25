from pathlib import Path
from typing import Union

import torch
from torch import nn


def load_weights(networks: {str: nn.Module}, model_file: Union[str, Path], *, strict: bool = True):
    weights = torch.load(model_file)
    for key, module in networks.items():
        if key is not None and key in weights:
            module.load_state_dict(weights[key], strict=strict)
    return networks
