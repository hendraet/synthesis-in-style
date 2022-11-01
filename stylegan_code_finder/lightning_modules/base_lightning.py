import os
import torch
from torch import nn
import pytorch_lightning as pl
from training_builder.base_train_builder import BaseTrainBuilder
from networks.trans_u_net.utils import DiceLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils.clamped_cosine import ClampedCosineAnnealingLR


class BaseSegmenter(pl.LightningModule):
    def __init__(self, training_builder: BaseTrainBuilder, configs):
        super().__init__()
        self.segmentation_network = training_builder.segmentation_network
        self.optimizers = list(training_builder.get_optimizers().values())
        self.configs = configs
        self.updater = training_builder.get_updater()
        self.iterations_per_epoch = self.get_iterations_per_epoch()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizers = self.optimizers
        schedulers = self.get_scheduler(optimizers)
        return optimizers, schedulers

    def get_scheduler(self, optimizers):
        if 'cosine_max_update_epoch' in self.configs:
            cosine_end_iteration = self.configs['cosine_max_update_epoch'] * self.iterations_per_epoch
        elif 'cosine_max_update_iter' in self.configs:
            cosine_end_iteration = self.configs['cosine_max_update_iter']
        else:
            cosine_end_iteration = self.configs['epochs']

        schedulers = []
        for optimizer in optimizers:
            if self.configs["warm_restarts"]:
                schedulers.append(CosineAnnealingWarmRestarts(optimizer, cosine_end_iteration,
                                                              eta_min=self.configs['end_lr']))
            else:
                schedulers.append(ClampedCosineAnnealingLR(optimizer, cosine_end_iteration,
                                                           eta_min=self.configs['end_lr']))
        return schedulers

    def get_iterations_per_epoch(self) -> int:
        num_iterations_in_epoch = self.updater.epoch_length - self.updater.iteration_in_epoch
        if 'max_iter' in self.configs:
            return min(self.configs['max_iter'] - self.updater.iteration, self.updater.epoch_length)
        else:
            return num_iterations_in_epoch
