import cv2
import numpy
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any

from utils.segmentation_utils import get_contours_from_prediction


class BaseSegmenter(nn.Module):
    """
    Abstract superclass that should enable easier handling of different segmentation models. New segmentation models
    should be implemented as subclass, overwrite the forward method as well as the predict methods (if needed).
    """

    def __init__(self, background_class_id: int = 0, min_confidence: float = 0.0, min_contour_area: int = 0,
                 num_input_channels: int = 3):
        super().__init__()
        self.background_class_id = background_class_id
        self.min_confidence = min_confidence
        self.min_contour_area = min_contour_area
        self.num_input_channels = num_input_channels

    def remove_too_small_contours(self, predictions: torch.Tensor) -> torch.Tensor:
        # following the DocUFCN paper all contours with area < self.min_contour_area shall be discarded
        clean_predictions = predictions.clone().cpu()
        for image_id in range(len(predictions)):
            class_predictions = predictions[image_id]
            for class_id, class_prediction in enumerate(class_predictions):
                if class_id == self.background_class_id:
                    continue

                contours = get_contours_from_prediction(class_prediction)
                if contours is None:
                    continue

                remove_mask = numpy.ones(class_prediction.shape, dtype=numpy.uint8)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < self.min_contour_area:
                        remove_mask = cv2.fillPoly(remove_mask, [contour], (0, 0, 0))
                clean_predictions[image_id, class_id] *= remove_mask

        return clean_predictions.to(predictions.device)

    def postprocess(self, predictions: torch.Tensor) -> torch.Tensor:
        processed_predictions = predictions.clone()
        confidence_not_high_enough = processed_predictions < self.min_confidence
        processed_predictions[confidence_not_high_enough] = 0
        processed_predictions = self.remove_too_small_contours(processed_predictions)
        return processed_predictions

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        network_output = self.forward(x)
        softmax_predictions = F.softmax(network_output, dim=1)
        return self.postprocess(softmax_predictions)

    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        predictions = self.predict(x)
        predicted_classes = torch.unsqueeze(torch.max(predictions, dim=1)[1], dim=1)
        return predicted_classes

    def forward(self, x: Any):
        raise NotImplementedError
