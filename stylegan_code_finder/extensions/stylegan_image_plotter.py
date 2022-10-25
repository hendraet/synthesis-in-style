from typing import List

import torch

from pytorch_training.extensions import ImagePlotter


class StyleGANImagePlotter(ImagePlotter):

    def get_predictions(self) -> List[torch.Tensor]:
        predictions = []
        for network in self.networks:
            predictions.append(network([self.input_images])[0])
        return predictions
