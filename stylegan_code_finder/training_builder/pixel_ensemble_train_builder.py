from pathlib import Path
from typing import Dict, Union

import torch
from pytorch_training import Updater
from pytorch_training.distributed.utils import strip_parallel_module
from pytorch_training.extensions import Snapshotter, ImagePlotter, Logger, Evaluator
from pytorch_training.optimizer import GradientClipAdam
from pytorch_training.triggers import get_trigger
from torch.optim import Optimizer

from evaluation.dataset_gan_evaluator import DatasetGANEvaluator, DiceGANEvalFunc
from networks.base_segmenter import BaseSegmenter
from networks.pixel_classifier.model import PixelEnsembleClassifier
from training_builder.base_train_builder import BaseTrainBuilder
from updater.dataset_gan_updater import DatasetGANUpdater
from visualization.dataset_gan_plotter import DatasetGANPlotter


class PixelEnsembleTrainBuilder(BaseTrainBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_segmentation_network()
        self.segmentation_network = self._prepare_segmentation_network(self.segmentation_network)
        self.optimizer_opts = {
            'betas': (self.config['beta1'], self.config['beta2']),
            'weight_decay': self.config['weight_decay'],
            'lr': float(self.config['lr']),
        }

    def _initialize_segmentation_network(self):
        try:
            feature_vector_length = self.train_data_loader.dataset.get_feature_vector_length()
        except AttributeError:
            raise RuntimeError('The given dataset does not seem to implement the "get_feature_vector_length" method. '
                               'However, this is required for initializing the PixelEnsemble classifier')

        ensemble = PixelEnsembleClassifier(self.config['numpy_class'], feature_vector_length, self.config['num_models'])
        self.segmentation_network = ensemble

    def _prepare_segmentation_network(self, segmentation_network, network_name: str = 'segmentation_network'):
        for sub_network_name, network in segmentation_network.get_networks().items():
            prepared_network = super()._prepare_segmentation_network(network, sub_network_name)
            segmentation_network.set_network(sub_network_name, prepared_network)
        return segmentation_network

    def get_networks_for_updater(self) -> Dict[str, BaseSegmenter]:
        return self.segmentation_network.get_networks()

    def get_optimizers(self) -> Dict[str, Optimizer]:
        optimizers = {}
        for i, sub_network in enumerate(self.segmentation_network.get_networks().values()):
            optimizer_name = f'optimizer_{i}'
            optimizers[optimizer_name] = GradientClipAdam(sub_network.parameters(), **self.optimizer_opts)
        return optimizers

    def get_updater(self) -> Updater:
        updater = DatasetGANUpdater(
            iterators={'feature_vectors': self.train_data_loader},
            networks=self.get_networks_for_updater(),
            optimizers=self.get_optimizers(),
            device='cuda',
            copy_to_device=(self.world_size == 1),
        )
        return updater

    def get_snapshotter(self) -> Union[Snapshotter, None]:
        if self.rank != 0:
            return None
        snapshotter = Snapshotter(
            {
                **{sub_network_name: strip_parallel_module(network) for sub_network_name, network in self.segmentation_network.get_networks().items()},
                **self.get_optimizers(),
            },
            self.config['log_dir'],
            trigger=get_trigger((self.config['snapshot_save_iter'], 'iteration'))
        )
        return snapshotter

    def get_evaluator(self, logger: Logger) -> Union[Evaluator, None]:
        if self.val_data_loader is None:
            print("No validation dataset is given. Omitting Evaluator...")
            return None
        evaluator = DatasetGANEvaluator(
            self.val_data_loader,
            logger,
            DiceGANEvalFunc(self.segmentation_network, self.rank),
            self.rank,
            dataset=self.val_data_loader.dataset,
            trigger=get_trigger((1, 'epoch')),
        )
        return evaluator

    def get_image_plotter(self) -> Union[ImagePlotter, None]:
        input_images, input_pixels, label_images = self.train_data_loader.dataset.get_images_for_plot(self.config['display_size'])

        label_images = torch.stack(label_images).unsqueeze(dim=1)

        image_plotter = DatasetGANPlotter(
            input_pixels,
            [],
            self.config["log_dir"],
            plot_dataset=self.train_data_loader.dataset,
            ensemble=self.segmentation_network,
            trigger=get_trigger((self.config['image_save_iter'], 'iteration')),
            plot_to_logger=True,
            label_images=label_images,
            images=torch.stack(input_images),
            class_to_color_map=Path(self.config['class_to_color_map']),
        )

        return image_plotter
