import json
import os
from pathlib import Path
from typing import Dict, Union

import numpy
import torch
import torch.nn.functional as F
from pytorch_training.data.json_dataset import JSONDataset
from pytorch_training.images import is_image

from utils.augment_dataset import augment_image
from utils.segmentation_utils import segmentation_image_to_class_image


class SegmentationDataset(JSONDataset):

    def __init__(self, *args, class_to_color_map_path: Path = None, background_class_name: str = 'background',
                 image_size: int = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.background_class_name = background_class_name
        self.image_size = image_size

        assert class_to_color_map_path is not None, "Segmentation Dataset requires a class_to_color_map"
        with class_to_color_map_path.open() as f:
            self.class_to_color_map = json.load(f)
            self.reversed_class_to_color_map = {v: k for k, v in self.class_to_color_map.items()}
        assert self.background_class_name in self.class_to_color_map, \
            f"Background class name: {self.background_class_name} not found in class to color map"

    def load_json_data(self, json_data: Union[dict, list]):
        self.image_data = json_data
        self.image_data = [path for file_path in self.image_data if is_image(path := file_path['file_name'])]

    def class_image_to_tensor(self, class_image: numpy.ndarray) -> torch.Tensor:
        class_image = class_image[numpy.newaxis, ...]
        class_image = torch.from_numpy(class_image)
        if self.image_size is not None:
            with torch.no_grad():
                class_image = F.interpolate(class_image[None, ...], (self.image_size, self.image_size))[0]
        return class_image

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.image_data[index]
        if self.root is not None:
            path = os.path.join(self.root, path)

        image = self.loader(path)
        input_image = image.crop((0, 0, image.width // 2, image.height))
        segmentation_image = image.crop((image.width // 2, 0, image.width, image.height))

        input_image = self.transforms(input_image)
        segmentation_image = segmentation_image_to_class_image(numpy.array(segmentation_image),
                                                               self.background_class_name, self.class_to_color_map)
        segmentation_image = self.class_image_to_tensor(segmentation_image).type(torch.LongTensor)
        assert input_image.shape[-2:] == segmentation_image.shape[-2:], "Input image and segmentation shape should " \
                                                                        "be the same!"

        return {
            "images": input_image,
            "segmented": segmentation_image
        }


class AugmentedSegmentationDataset(SegmentationDataset):
    """
    The original dataset will be inflated by the number of augmented images (num_augmentations). Iterating over the
    whole dataset will return the original image exactly once and the number of augmented images otherwise.
    """

    def __init__(self, *args, **kwargs):
        self.num_augmentations = kwargs.pop("num_augmentations")
        assert isinstance(self.num_augmentations, int), "num_augmentations must be an Integer"
        super().__init__(*args, **kwargs)

    def __len__(self) -> int:
        return self.num_augmentations * super().__len__()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Dataset was artificially inflated, so we have to shift the index back
        original_dataset_length = super().__len__()
        actual_index = index % original_dataset_length
        path = self.image_data[actual_index]

        if self.root is not None:
            path = os.path.join(self.root, path)

        image = self.loader(path)
        input_image = image.crop((0, 0, image.width // 2, image.height))
        segmentation_image = image.crop((image.width // 2, 0, image.width, image.height))

        if index // original_dataset_length != 0:
            # After first iteration through the original dataset, return augmented images
            input_image, segmentation_image = augment_image(input_image, segmentation_image, num_images=1)[0]

        input_image = self.transforms(input_image)
        segmentation_image = segmentation_image_to_class_image(numpy.array(segmentation_image),
                                                               self.background_class_name, self.class_to_color_map)
        segmentation_image = self.class_image_to_tensor(segmentation_image).type(torch.LongTensor)
        assert input_image.shape[-2:] == segmentation_image.shape[-2:], "Input image and segmentation shape should " \
                                                                        "be the same!"

        return {
            "images": input_image,
            "segmented": segmentation_image
        }
