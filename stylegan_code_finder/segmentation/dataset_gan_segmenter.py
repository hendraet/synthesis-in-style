from typing import List, Dict, Tuple

import numpy
import torch
from torch import nn

from data.dataset_gan_dataset import scale_activations
from networks.pixel_classifier.model import PixelClassifier, PixelEnsembleClassifier
from segmentation.base_dataset_segmenter import BaseDatasetSegmenter


class DatasetGANSegmenter(BaseDatasetSegmenter):
    """
    This Segmenter makes use of a model (PixelEnsembleClassifier) that directly classifies the activations of a
    StyleGAN model.
    """
    def __init__(self, *args, classifier_path: str, feature_size: int, upsamplers: List[nn.Upsample], **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble = self.load_ensemble(classifier_path, feature_size)
        self.upsamplers = upsamplers

    def load_ensemble(self, path: str, feature_size: int) -> PixelEnsembleClassifier:
        ensemble = PixelEnsembleClassifier(len(self.class_to_color_map.keys()), self.image_size, 0)
        checkpoint = torch.load(path)
        for key in checkpoint.keys():
            if "network" in key and "optimizer" not in key:
                model = PixelClassifier(len(self.class_to_color_map.keys()), feature_size)
                model.load_state_dict(checkpoint[key])
                model.cuda()
                model.eval()
                ensemble.add_network(model)
        return ensemble

    @torch.no_grad()
    def predict_labels(self, activations: torch.Tensor) -> torch.Tensor:
        b, w, h, f = activations.shape
        activations_batch = activations.reshape([b * w * h, f])
        labels = self.ensemble.predict_classes(activations_batch)
        label_images = labels.reshape([b, self.image_size, self.image_size])

        return label_images

    def label_images_to_color_images(self, label_images: torch.Tensor) -> numpy.ndarray:
        batch_size, _, height, width = label_images.shape
        color_images = numpy.zeros((batch_size, height, width, 3), dtype='uint8')
        color_images[:, :, :] = self.class_to_color_map['background']

        for class_id, (class_name, color) in enumerate(self.class_to_color_map.items()):
            if class_name == 'background':
                continue
            class_mask = (label_images == class_id).cpu().numpy().squeeze()
            color_images[class_mask] = color
        return color_images

    def create_segmentation_image(self, activations: Dict[int, torch.Tensor]) -> Tuple[numpy.ndarray, List[int]]:
        scaled_activations = scale_activations([activations], self.upsamplers)[0]
        label_images = torch.unsqueeze(self.predict_labels(scaled_activations), dim=1)
        segmentation_images = self.label_images_to_color_images(label_images)

        return segmentation_images, []
