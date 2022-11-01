import os
import torch
from torch import nn
import pytorch_lightning as pl
from training_builder.trans_u_net_train_builder import TransUNetTrainBuilder
from networks.trans_u_net.utils import DiceLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils.clamped_cosine import ClampedCosineAnnealingLR


class TransUNetSegmenter(pl.LightningModule):
    def __init__(self, u_net_train_builder: TransUNetTrainBuilder, configs):
        super().__init__()
        self.segmentation_network = u_net_train_builder.segmentation_network
        self.u_net_optimizers = list(u_net_train_builder.get_optimizers().values())
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(configs['num_classes'])
        self.configs = configs
        self.updater = u_net_train_builder.get_updater()
        self.iterations_per_epoch = self.get_iterations_per_epoch()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        prediction = self.segmentation_network(batch['images'])

        ground_truth = torch.squeeze(batch['segmented'], dim=1)
        loss_ce = self.ce_loss(prediction, ground_truth.long())
        loss_dice = self.dice_loss(prediction, ground_truth, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        prediction = self.segmentation_network(batch['images'])

        ground_truth = torch.squeeze(batch['segmented'], dim=1)
        loss_ce = self.ce_loss(prediction, ground_truth.long())
        loss_dice = self.dice_loss(prediction, ground_truth, softmax=True)
        val_loss = 0.5 * loss_ce + 0.5 * loss_dice
        self.log('dice_val_loss', loss_dice)
        self.log('ce_val_loss', loss_ce)
        self.log('val_loss', val_loss)

    def configure_optimizers(self):
        optimizers = self.u_net_optimizers
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
