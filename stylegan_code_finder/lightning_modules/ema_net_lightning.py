import os
import torch
from torch import nn
import pytorch_lightning as pl
from training_builder.ema_net_train_builder import EMANetTrainBuilder
from networks.trans_u_net.utils import DiceLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils.clamped_cosine import ClampedCosineAnnealingLR
from lightning_modules.base_lightning import BaseSegmenter


class EmaNetSegmenter(BaseSegmenter):
    def __init__(self, ema_net_train_builder: EMANetTrainBuilder, configs, segmentation_plotter, wandb_logger):
        super().__init__(ema_net_train_builder, configs, segmentation_plotter, wandb_logger)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(configs['num_classes'])
        self.em_mom = configs['em_mom']

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
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
        # this is the validation loop
        return 0
