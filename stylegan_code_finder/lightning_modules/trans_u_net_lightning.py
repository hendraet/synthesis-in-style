import os
import torch
from torch import nn
from training_builder.trans_u_net_train_builder import TransUNetTrainBuilder
from networks.trans_u_net.utils import DiceLoss
from lightning_modules.base_lightning import BaseSegmenter


class TransUNetSegmenter(BaseSegmenter):
    def __init__(self, u_net_train_builder: TransUNetTrainBuilder, configs):
        super().__init__(u_net_train_builder, configs)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(configs['num_classes'])

    def training_step(self, batch, batch_idx):
        prediction = self.segmentation_network(batch['images'])

        ground_truth = torch.squeeze(batch['segmented'], dim=1)
        loss_ce = self.ce_loss(prediction, ground_truth.long())
        loss_dice = self.dice_loss(prediction, ground_truth, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        super(TransUNetSegmenter, self).validation_step(batch, batch_idx)
        prediction = self.segmentation_network(batch['images'])

        ground_truth = torch.squeeze(batch['segmented'], dim=1)
        loss_ce = self.ce_loss(prediction, ground_truth.long())
        loss_dice = self.dice_loss(prediction, ground_truth, softmax=True)
        val_loss = 0.5 * loss_ce + 0.5 * loss_dice
        self.log('dice_val_loss', loss_dice)
        self.log('ce_val_loss', loss_ce)
        self.log('val_loss', val_loss)
