from typing import Dict

from pytorch_training import Updater
from torch.optim import Optimizer, SGD

from networks.base_segmenter import BaseSegmenter
from networks.ema_net.bn_lib.nn import patch_replication_callback
from networks.ema_net.network import EMANet
from networks.ema_net.utils import get_params
from training_builder.base_train_builder import BaseSingleNetworkTrainBuilder
from updater.segmentation_updater import EMANetUpdater


class EMANetTrainBuilder(BaseSingleNetworkTrainBuilder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_segmentation_network()
        self.segmentation_network = self._prepare_segmentation_network(self.segmentation_network)

    def _initialize_segmentation_network(self):
        use_pretrained_resnet = True if self.config['fine_tune'] is None else False
        segmentation_network = EMANet(self.config['num_classes'], self.config['n_layers'],
                                      use_pretrained_resnet=use_pretrained_resnet)
        self.segmentation_network = segmentation_network

    def _prepare_segmentation_network(self, segmentation_network: BaseSegmenter,
                                      network_name: str = 'segmentation_network') -> BaseSegmenter:
        segmentation_network = super()._prepare_segmentation_network(segmentation_network)
        if self.world_size > 1:
            patch_replication_callback(segmentation_network)
        return segmentation_network

    def get_optimizers(self) -> Dict[str, Optimizer]:
        optimizer = SGD(
            params=[
                {
                    'params': get_params(self.segmentation_network, key='1x'),
                    'lr': self.config['lr'],
                    'weight_decay': self.config['weight_decay'],
                },
                {
                    'params': get_params(self.segmentation_network, key='1y'),
                    'lr': self.config['lr'],
                    'weight_decay': 0,
                },
                {
                    'params': get_params(self.segmentation_network, key='2x'),
                    'lr': 2 * self.config['lr'],
                    'weight_decay': 0.0,
                }
            ],
            momentum=self.config['lr_mom']
        )
        return {'main': optimizer}

    def get_updater(self) -> Updater:
        updater = EMANetUpdater(
            em_mom=self.config['em_mom'],
            iterators={'images': self.train_data_loader},
            networks=self.get_networks_for_updater(),
            optimizers=self.get_optimizers(),
            device='cuda',
            copy_to_device=(self.world_size == 1)
        )
        return updater
