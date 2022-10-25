from typing import List, Type

import torch
from torch.utils.data import Dataset

from networks.base_segmenter import BaseSegmenter
from visualization.segmentation_plotter import SegmentationPlotter


class DatasetGANPlotter(SegmentationPlotter):
    def __init__(self, *args, plot_dataset: Dataset, ensemble: Type[BaseSegmenter], images: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_dataset = plot_dataset
        self.ensemble = ensemble

        # To make sure we can use the standard ImagePlotter run method
        self.networks = self.ensemble.networks.values()

        assert len(images.shape) == 4 and images.shape[1] in (1, 3), \
            "Ground truth images should have the shape (batch size, [1|3], height, width), where the channel " \
            "dimension should either be 1 or 3 "
        # Since label images and network outputs will be in normalized form, the images have to be normalized as well
        normalized_images = images * 2 - 1
        self.real_images = normalized_images

    def get_predictions(self) -> List[torch.Tensor]:
        predictions = [self.real_images, self.label_images]

        with torch.no_grad():
            result_images = []
            # input_images might be a misleading name since these "images" are actually activations. However,
            # they work analogously to normal images.
            for image in self.input_images:
                predicted_classes = []
                for row_ind, row in enumerate(image):
                    pixel_label = self.ensemble.predict_classes(row)
                    predicted_classes.append(pixel_label.cpu())

                result_images.append(torch.stack(predicted_classes))

        network_outputs = torch.stack(result_images).squeeze(dim=-1).unsqueeze(dim=1)
        network_outputs = self.label_images_to_color_images(network_outputs)

        predictions.append(network_outputs)
        return predictions
