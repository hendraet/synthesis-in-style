from typing import List

import torch
from lightning_modules.base_lightning import BaseSegmenter
from networks.doc_ufcn.doc_ufcn import DocUFCN
from pytorch_training.optimizer import GradientClipAdam
from torch import nn
from torch.optim import Optimizer


class DocUFCNSegmenter(BaseSegmenter):
    def __init__(self, configs: dict):
        super().__init__(configs)
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor(configs['class_weights']))
        self.optimizers = self.get_optimizers()

    def _initialize_segmentation_network(self):
        segmentation_network_class = DocUFCN
        self.segmentation_network = segmentation_network_class(3, 3)

    def training_step(self, batch, batch_idx):
        segmentation_prediction = self.segmentation_network(batch['images'])
        batch_size, num_classes, height, width = segmentation_prediction.shape
        segmentation_prediction = segmentation_prediction.permute(0, 2, 3, 1)
        segmentation_prediction = torch.reshape(segmentation_prediction, (-1, num_classes))

        label_image = batch['segmented']
        label_image = label_image.permute(0, 2, 3, 1)
        label_image = label_image.reshape((-1,))

        loss = self.ce_loss(segmentation_prediction, label_image)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        segmentation_prediction = self.segmentation_network(batch['images'])
        batch_size, num_classes, height, width = segmentation_prediction.shape
        segmentation_prediction = segmentation_prediction.permute(0, 2, 3, 1)
        segmentation_prediction = torch.reshape(segmentation_prediction, (-1, num_classes))

        label_image = batch['segmented']
        label_image = label_image.permute(0, 2, 3, 1)
        label_image = label_image.reshape((-1,))

        loss = self.ce_loss(segmentation_prediction, label_image)
        self.log('val_loss', loss)

    def get_optimizers(self) -> List[Optimizer]:
        optimizer_opts = {
            'betas': (self.configs['beta1'], self.configs['beta2']),
            'weight_decay': self.configs['weight_decay'],
            'lr': float(self.configs['lr']),
        }
        optimizer = GradientClipAdam(self.segmentation_network.parameters(), **optimizer_opts)
        return [optimizer]
