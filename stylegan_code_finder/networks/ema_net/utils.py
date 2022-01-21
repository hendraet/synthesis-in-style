from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm


# This file was extracted from https://github.com/XiaLiPKU/EMANet

def get_params(model, key):
    if key == '1x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                yield m[1].weight
    if key == '1y':
        for m in model.named_modules():
            if isinstance(m[1], _BatchNorm):
                if m[1].weight is not None:
                    yield m[1].weight
    if key == '2x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], _BatchNorm):
                if m[1].bias is not None:
                    yield m[1].bias
