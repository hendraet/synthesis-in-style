from typing import List

import numpy as np
import torch
from lightning_modules.base_lightning import BaseSegmenter
from networks.trans_u_net.utils import DiceLoss, TverskyLoss
from networks.trans_u_net.vit_seg_modeling import VisionTransformer, VIT_CONFIGS
from torch import nn
from torch.optim import Optimizer, SGD


class TransUNetSegmenter(BaseSegmenter):
    def __init__(self, configs: dict):
        super().__init__(configs)
        #self.ce_loss = nn.CrossEntropyLoss()
        #self.dice_loss = DiceLoss(configs['num_classes'])
        self.optimizers = self.get_optimizers()
        self.tversky_loss = TverskyLoss(configs['num_classes'], 0.3, 0.7)

    def _initialize_segmentation_network(self):
        transformer_config = VIT_CONFIGS[self.configs['pretrained_model_name']]
        transformer_config.n_classes = self.configs['num_classes']
        transformer_config.n_skip = self.configs['num_skip_channels']
        vit_patches_size = self.configs['vit_patch_size']
        transformer_config.patches.grid = (self.configs['image_size'] // vit_patches_size,
                                           self.configs['image_size'] // vit_patches_size)

        segmentation_network = VisionTransformer(transformer_config, img_size=self.configs['image_size'],
                                                 num_classes=transformer_config.n_classes)
        if self.configs['fine_tune'] is None:
            segmentation_network.load_from(weights=np.load(self.configs['pretrained_path']))
        self.segmentation_network = segmentation_network

    def training_step(self, batch, batch_idx):
        prediction = self.segmentation_network(batch['images'])

        ground_truth = torch.squeeze(batch['segmented'], dim=1)
        #loss_ce = self.ce_loss(prediction, ground_truth.long())
        #loss_dice = self.dice_loss(prediction, ground_truth, softmax=True)
        #loss = 0.5 * loss_ce + 0.5 * loss_dice
        loss = self.tversky_loss(prediction, ground_truth, softmax=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        prediction = self.segmentation_network(batch['images'])

        ground_truth = torch.squeeze(batch['segmented'], dim=1)
        #loss_ce = self.ce_loss(prediction, ground_truth.long())
        #loss_dice = self.dice_loss(prediction, ground_truth, softmax=True)
        #val_loss = 0.5 * loss_ce + 0.5 * loss_dice
        val_loss = self.tversky_loss(prediction, ground_truth, softmax=True)
        #self.log('val_dice_loss', loss_dice)
        #self.log('val_ce_loss', loss_ce)
        self.log('val_loss', val_loss)
        self.log_precision_recall_accuracy(prediction, ground_truth, softmax=True)
        self.log_handwriting_precision_recall_accuracy(prediction, ground_truth, softmax=True)

    def get_optimizers(self) -> List[Optimizer]:
        optimizer_opts = {
            'lr': self.configs['lr'],
            'momentum': self.configs['momentum'],
            'weight_decay': self.configs['weight_decay']
        }
        optimizer = SGD(self.segmentation_network.parameters(), **optimizer_opts)
        return [optimizer]
