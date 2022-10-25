import json
from pathlib import Path
from typing import List

import numpy
import torch
from PIL import ImageColor
from pytorch_training.extensions import ImagePlotter


class SegmentationPlotter(ImagePlotter):

    def __init__(self, *args, class_to_color_map: Path = None, label_images: torch.Tensor = None,  **kwargs):
        super().__init__(*args, **kwargs)
        assert class_to_color_map is not None, "Class to Color map must be supplied to SegmentationPlotter!"
        with class_to_color_map.open() as f:
            self.class_to_color_map = json.load(f)
        assert label_images is not None, "Label images must be supplied to SegmentationPlotter!"
        assert len(label_images.shape) == 4, "Label images should have the shape (batch size, channels, height, width)"
        self.label_images = self.label_images_to_color_images(label_images)

    def label_images_to_color_images(self, label_images: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = label_images.shape
        color_images = numpy.zeros((batch_size, height, width, 3), dtype='uint8')
        color_images[:, :, :] = ImageColor.getrgb(self.class_to_color_map['background'])

        for class_id, (class_name, color) in enumerate(self.class_to_color_map.items()):
            if class_name == 'background':
                continue
            class_mask = (label_images == class_id).cpu().numpy().squeeze()
            color_images[class_mask] = ImageColor.getrgb(color)
        color_images = (color_images / 255) * 2 - 1
        return torch.from_numpy(color_images.transpose(0, 3, 1, 2)).to(label_images.device)

    def get_predictions(self) -> List[torch.Tensor]:
        predictions = [self.input_images, self.label_images]
        assert len(self.networks) == 1, "Segmentation Plotter only supports 1 network!"
        with torch.no_grad():
            predicted_classes = self.networks[0].predict_classes(self.input_images)
        network_output = self.label_images_to_color_images(predicted_classes)
        predictions.append(network_output)
        return predictions
