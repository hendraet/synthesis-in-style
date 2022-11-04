import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils.clamped_cosine import ClampedCosineAnnealingLR
from visualization.segmentation_plotter_lightning import SegmentationPlotter
from networks import load_weights


class BaseSegmenter(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.segmentation_network = None
        self._initialize_segmentation_network()
        if configs['fine_tune'] is not None:
            load_weights(self.segmentation_network, configs['fine_tune'], key='segmentation_network')
        self.optimizers = None
        self.iterations_per_epoch = self.get_iterations_per_epoch()
        self.segmentation_plotter = SegmentationPlotter(configs)
        self.num_val_visualization = configs['num_val_visualization']

    def _initialize_segmentation_network(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        if batch_idx < self.num_val_visualization:
            image_dest = self.segmentation_plotter.run(self.segmentation_network, batch)
            self.logger.log_image(key="samples", images=[image_dest])

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
        if 'max_iter' in self.configs:
            return min(self.configs['max_iter'], self.configs['num_iter_epoch'])
        else:
            return self.configs['num_iter_epoch']
