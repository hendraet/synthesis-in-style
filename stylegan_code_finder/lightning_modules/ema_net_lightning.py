import torch
from training_builder.ema_net_train_builder import EMANetTrainBuilder
from lightning_modules.base_lightning import BaseSegmenter


class EmaNetSegmenter(BaseSegmenter):
    def __init__(self, ema_net_train_builder: EMANetTrainBuilder, configs):
        super().__init__(ema_net_train_builder, configs)
        self.em_mom = configs['em_mom']

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
