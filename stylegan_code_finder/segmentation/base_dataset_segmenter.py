import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy
import torch
import torch.nn.functional as F
from PIL import ImageColor

import global_config
from utils.segmentation_utils import PredictedClusters


class BaseDatasetSegmenter:

    def __init__(self, base_dir: Path, image_size: int, class_to_color_map: Dict):
        self.base_dir = base_dir
        self.image_size = image_size
        self.debug = global_config.debug
        self.debug_images = {}
        self.max_debug_text_size = 20
        self.class_to_color_map = self.load_class_to_color_map(class_to_color_map)
        self.class_id_map = self.build_class_id_map(self.class_to_color_map)

    def load_class_to_color_map(self, class_to_color_map: dict) -> dict:
        return {class_name: ImageColor.getrgb(color) for class_name, color in class_to_color_map.items()}

    def build_class_id_map(self, class_to_color_map: dict) -> dict:
        return {class_name: class_id for class_id, class_name in enumerate(class_to_color_map)}

    def resize_to_image_size(self, tensors: PredictedClusters) -> PredictedClusters:
        resized = {}
        for key, class_tensors in tensors.items():
            resized_class_tensors = {}
            for class_name, tensor in class_tensors.items():
                if tensor.shape[-1] < self.image_size:
                    tensor = F.interpolate(tensor[:, None, ...].type(torch.uint8),
                                           (self.image_size, self.image_size)).type(tensor.dtype).squeeze(1)
                resized_class_tensors[class_name] = tensor
            resized[key] = resized_class_tensors
        return resized

    def prepare_image_segmentation(self, activations, class_label_map):
        if self.debug:
            self.debug_images.clear()

        predicted_clusters = self.predict_clusters(activations, class_label_map)
        predicted_clusters = self.resize_to_image_size(predicted_clusters)
        return predicted_clusters

    @staticmethod
    def dilate_image(image: numpy.ndarray, kernel: numpy.ndarray = None, kernel_size: int = 3) -> numpy.ndarray:
        if kernel is None:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size)).astype(numpy.uint8)

        return cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)

        # kernel = torch.from_numpy(kernel[numpy.newaxis, numpy.newaxis, ...]).to(predicted_clusters.device)
        # dilated_slices = [
        #     torch.clamp(F.conv2d(class_map, kernel, padding=(1, 1)), 0, 1)
        #     for class_map in torch.split(predicted_clusters.type(torch.float32), 1, dim=1)
        # ]
        # dilated = torch.cat(dilated_slices, dim=1)
        # return dilated

    def create_segmentation_image(self, activations: Dict[int, torch.Tensor]) -> Tuple[numpy.ndarray, List[int]]:
        raise NotImplementedError
