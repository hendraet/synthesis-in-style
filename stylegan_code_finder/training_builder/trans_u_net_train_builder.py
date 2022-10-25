from typing import Dict

import numpy
from pytorch_training import Updater
from torch.optim import Optimizer, SGD

from networks.trans_u_net.vit_seg_modeling import VisionTransformer, VIT_CONFIGS
from training_builder.base_train_builder import BaseSingleNetworkTrainBuilder
from updater.segmentation_updater import TransUNetUpdater


class TransUNetTrainBuilder(BaseSingleNetworkTrainBuilder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_segmentation_network()
        self.segmentation_network = self._prepare_segmentation_network(self.segmentation_network)
        self.optimizer_opts = {
            'lr': self.config['lr'],
            'momentum': self.config['momentum'],
            'weight_decay': self.config['weight_decay']
        }

    def _initialize_segmentation_network(self):
        transformer_config = VIT_CONFIGS[self.config['pretrained_model_name']]
        transformer_config.n_classes = self.config['num_classes']
        transformer_config.n_skip = self.config['num_skip_channels']
        vit_patches_size = self.config['vit_patch_size']
        transformer_config.patches.grid = (self.config['image_size'] // vit_patches_size,
                                           self.config['image_size'] // vit_patches_size)

        segmentation_network = VisionTransformer(transformer_config, img_size=self.config['image_size'],
                                                 num_classes=transformer_config.n_classes)
        if self.config['fine_tune'] is None:
            segmentation_network.load_from(weights=numpy.load(self.config['pretrained_path']))
        self.segmentation_network = segmentation_network

    def get_optimizers(self) -> Dict[str, Optimizer]:
        optimizer = SGD(self.segmentation_network.parameters(), **self.optimizer_opts)
        return {'main': optimizer}

    def get_updater(self) -> Updater:
        updater = TransUNetUpdater(
            num_classes=self.config['num_classes'],
            iterators={'images': self.train_data_loader},
            networks=self.get_networks_for_updater(),
            optimizers=self.get_optimizers(),
            device=self.rank,
            copy_to_device=(self.world_size == 1)
        )
        return updater
