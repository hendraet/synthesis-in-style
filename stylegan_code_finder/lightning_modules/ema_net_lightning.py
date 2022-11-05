import torch
from lightning_modules.base_lightning import BaseSegmenter
from networks.ema_net.network import EMANet
from torch.optim import Optimizer, SGD
from networks.ema_net.utils import get_params
from typing import List


class EmaNetSegmenter(BaseSegmenter):
    def __init__(self, configs: dict):
        super().__init__(configs)
        self.em_mom = configs['em_mom']
        self.optimizers = self.get_optimizers()

    def _initialize_segmentation_network(self):
        use_pretrained_resnet = True if self.configs['fine_tune'] is None else False
        segmentation_network = EMANet(self.configs['num_classes'], self.configs['n_layers'],
                                      use_pretrained_resnet=use_pretrained_resnet,
                                      pretrained_path=self.configs['pretrained_path'] if use_pretrained_resnet else None)
        self.segmentation_network = segmentation_network

    def training_step(self, batch, batch_idx):
        self.segmentation_network.train()
        loss, mu = self.segmentation_network(batch['images'], torch.squeeze(batch['segmented'], dim=1))

        with torch.no_grad():
            mu = mu.mean(dim=0, keepdim=True)
            try:
                self.segmentation_network.emau.mu *= self.em_mom
                self.segmentation_network.emau.mu += mu * (1 - self.em_mom)
            except AttributeError:
                # TransUNet was not designed to work with DistributedDataParallel, so we need this hack to make it work.
                # It is generally not advised to change parameters of a DDP model, because it can mess with gradients.
                # However, in this gradients are not required anyways, so it doesn't matter.
                self.segmentation_network.module.emau.mu *= self.em_mom
                self.segmentation_network.module.emau.mu += mu * (1 - self.em_mom)

        loss = loss.mean()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        super(EmaNetSegmenter, self).validation_step(batch, batch_idx)

    def get_optimizers(self) -> List[Optimizer]:
        optimizer = SGD(
            params=[
                {
                    'params': get_params(self.segmentation_network, key='1x'),
                    'lr': self.configs['lr'],
                    'weight_decay': self.configs['weight_decay'],
                },
                {
                    'params': get_params(self.segmentation_network, key='1y'),
                    'lr': self.configs['lr'],
                    'weight_decay': 0,
                },
                {
                    'params': get_params(self.segmentation_network, key='2x'),
                    'lr': 2 * self.configs['lr'],
                    'weight_decay': 0.0,
                }
            ],
            momentum=self.configs['lr_mom']
        )
        return [optimizer]
