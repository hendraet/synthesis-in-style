from typing import List, Dict

import numpy
import torch
from torch.nn import Upsample
from tqdm import tqdm

from data.base_dataset_gan_dataset import BaseDatasetGANDataset
from utils.segmentation_utils import segmentation_image_to_class_image


def scale_activations(activations: List[Dict[int, torch.Tensor]], upsamplers: List[Upsample]) -> List[torch.Tensor]:
    scaled_activations = []
    for entry_activation in activations:
        assert len(entry_activation) == len(upsamplers), \
            f"uneven size of activations {len(entry_activation)} to upsamplers {len(upsamplers)}"

        batch_size = entry_activation[0].shape[0]
        image_size = entry_activation[0].shape[2] * int(upsamplers[0].scale_factor)
        feature_size = sum([e.shape[1] for e in entry_activation.values()])

        image_activations = torch.empty((batch_size, image_size, image_size, feature_size), device='cuda')

        feature_index = 0
        for idx, activation in entry_activation.items():
            upscaled_feature_maps = upsamplers[idx](activation.cuda())
            upscaled_feature_maps = upscaled_feature_maps.squeeze()

            new_index = feature_index + upscaled_feature_maps.shape[1]
            image_activations[:, :, :, feature_index:new_index] = torch.moveaxis(upscaled_feature_maps, 1, -1)
            feature_index = new_index

        scaled_activations.append(image_activations)
    return scaled_activations


class DatasetGANDataset(BaseDatasetGANDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_tensors(["activations"])
        self.load_data()
        self.create_sampling_buckets()

        if self.random_sampling:
            self.create_sampling_buckets()

    def load_data(self):
        assert self.activations is not None, "Activations were not loaded properly"
        activations = []
        for entry in tqdm(self.json_data, desc="Loading data", leave=False):
            self.image_paths.append(self.dataset_path / entry["image"])
            label_string = str(self.dataset_path / entry["label"])
            label_image = self.loader(label_string)
            label_array = segmentation_image_to_class_image(numpy.array(label_image), self.background_class_name,
                                                            self.class_to_color_map)
            label_array = self.class_image_to_tensor(label_array).type(torch.LongTensor)
            activations.append(self.activations[entry["activations"]])

            self.pixel_labels.append(label_array)
        del self.activations
        pixel_activations = scale_activations(activations, self.upsamplers)
        self.pixel_activations = [activation.cpu().numpy() for activation in pixel_activations]
        self.feature_vector_length = len(self.pixel_activations[0][0][0])
        self.pixel_labels = numpy.array(self.pixel_labels)
