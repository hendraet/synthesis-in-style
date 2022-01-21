from contextlib import contextmanager
from typing import Any, Dict

import numpy as np
import scipy
import scipy.stats
import torch
from torch import nn

from networks.base_segmenter import BaseSegmenter


class PixelEnsembleClassifier(BaseSegmenter):
    def __init__(self, numpy_class: int, dim: int, number_of_models: int):
        super().__init__()
        self.number_of_models = number_of_models
        self.networks = {}
        self.last_net_id = 0
        for i in range(self.number_of_models):
            self.networks["network_{}".format(i)] = PixelClassifier(numpy_class, dim)
            self.networks["network_{}".format(i)].init_weights()
            self.last_net_id += 1

    def get_networks(self) -> Dict[str, BaseSegmenter]:
        return self.networks

    def set_network(self, network_name: str, network: BaseSegmenter):
        self.networks[network_name] = network

    def add_network(self, network: BaseSegmenter):
        self.last_net_id += 1
        self.networks["network_{}".format(self.last_net_id)] = network

    def forward(self, x: Any):
        raise NotImplementedError

    def predict(self, x: Any):
        raise NotImplementedError

    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        predictions = torch.zeros((x.shape[0], len(self.networks)), device='cuda')

        i = 0
        for key, model in self.networks.items():
            predictions[:, i] = model.predict_classes(x).squeeze()
            i += 1

        pixel_class = torch.mode(predictions).values
        return pixel_class


@contextmanager
def ensemble_eval_mode(ensemble: PixelEnsembleClassifier):
    for network in ensemble.networks.values():
        network.eval()
    yield
    for network in ensemble.networks.values():
        network.train()


class PixelClassifier(BaseSegmenter):
    def __init__(self, numpy_class: int, dim: int):
        super().__init__()
        if numpy_class < 32:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class),
                # nn.Sigmoid()
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class),
                # nn.Sigmoid()
            )

    def init_weights(self, init_type: str = 'normal', gain: float = 0.02):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        """

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        network_output = self.forward(x)
        softmax_predictions = torch.log_softmax(network_output, dim=1)
        return softmax_predictions
