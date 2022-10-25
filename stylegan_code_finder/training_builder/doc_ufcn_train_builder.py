from typing import Dict

import torch
from pytorch_training import Updater
from pytorch_training.optimizer import GradientClipAdam
from torch.optim import Optimizer

from networks.doc_ufcn import get_doc_ufcn
from training_builder.base_train_builder import BaseSingleNetworkTrainBuilder
from updater.segmentation_updater import StandardUpdater


class DocUFCNTrainBuilder(BaseSingleNetworkTrainBuilder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_segmentation_network()
        self.segmentation_network = self._prepare_segmentation_network(self.segmentation_network)
        self.optimizer_opts = {
            'betas': (self.config['beta1'], self.config['beta2']),
            'weight_decay': self.config['weight_decay'],
            'lr': float(self.config['lr']),
        }

    def _initialize_segmentation_network(self):
        segmentation_network_class = get_doc_ufcn('base')
        self.segmentation_network = segmentation_network_class(3, 3)

    def get_optimizers(self) -> Dict[str, Optimizer]:
        optimizer = GradientClipAdam(self.segmentation_network.parameters(), **self.optimizer_opts)
        return {'main': optimizer}

    def get_updater(self) -> Updater:
        updater = StandardUpdater(
            iterators={'images': self.train_data_loader},
            networks=self.get_networks_for_updater(),
            optimizers=self.get_optimizers(),
            device=self.rank,
            class_weights=torch.Tensor(self.config['class_weights']).to(self.rank),
            copy_to_device=(self.world_size == 1)
        )
        return updater
