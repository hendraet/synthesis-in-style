import os
import torch
from torch import nn
import pytorch_lightning as pl
from training_builder.doc_ufcn_train_builder import DocUFCNTrainBuilder
from networks.trans_u_net.utils import DiceLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils.clamped_cosine import ClampedCosineAnnealingLR
from lightning_modules.base_lightning import BaseSegmenter


class DocUFCNSegmenter(BaseSegmenter):
    def __init__(self, docUFCN_train_builder: DocUFCNTrainBuilder, configs, segmentation_plotter, wandb_logger):
        super().__init__(docUFCN_train_builder, configs, segmentation_plotter, wandb_logger)
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor(configs['class_weights']))

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        segmentation_prediction = self.segmentation_network(batch['images'])
        batch_size, num_classes, height, width = segmentation_prediction.shape
        segmentation_prediction = segmentation_prediction.permute(0, 2, 3, 1)
        segmentation_prediction = torch.reshape(segmentation_prediction, (batch_size * height * width, num_classes))

        label_image = batch['segmented']
        label_image = label_image.permute(0, 2, 3, 1)
        label_image = label_image.reshape((-1,))

        loss = self.ce_loss(segmentation_prediction, label_image)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        segmentation_prediction = self.segmentation_network(batch['images'])
        batch_size, num_classes, height, width = segmentation_prediction.shape
        segmentation_prediction = segmentation_prediction.permute(0, 2, 3, 1)
        segmentation_prediction = torch.reshape(segmentation_prediction, (batch_size * height * width, num_classes))

        label_image = batch['segmented']
        label_image = label_image.permute(0, 2, 3, 1)
        label_image = label_image.reshape((-1,))

        loss = self.ce_loss(segmentation_prediction, label_image)
        self.log('val_loss', loss)

