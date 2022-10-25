from pathlib import Path
from typing import Union, Tuple, List, Dict

import numpy
import numpy as np
import torch
import torchvision.transforms
from numpy.random import default_rng
from torch import nn

from data.segmentation_dataset import SegmentationDataset


class BaseDatasetGANDataset(SegmentationDataset):
    def __init__(self, *args, tensor_path: str, upsample_mode: str,
                 class_probabilities: Union[float, List[float]] = 0.5, random_sampling=False, **kwargs):
        self.json_data = None
        super().__init__(*args, **kwargs)

        self.image_paths = []
        self.activations = None
        self.init_vectors = None
        self.sampling_buckets = []
        self.pixel_activations = None
        self.pixel_labels = []
        self.feature_vector_length = -1

        self.tensor_path = tensor_path
        self.dataset_path = Path(tensor_path).parent
        self.random_sampling = random_sampling

        if type(class_probabilities) is float:
            self.class_probabilities = [class_probabilities, 1 - class_probabilities]
        else:
            self.class_probabilities = class_probabilities

        mode = upsample_mode
        self.upsamplers = [
            nn.Upsample(scale_factor=self.image_size / 4, mode=mode),
            nn.Upsample(scale_factor=self.image_size / 4, mode=mode),
            nn.Upsample(scale_factor=self.image_size / 8, mode=mode),
            nn.Upsample(scale_factor=self.image_size / 8, mode=mode),
            nn.Upsample(scale_factor=self.image_size / 16, mode=mode),
            nn.Upsample(scale_factor=self.image_size / 16, mode=mode),
            nn.Upsample(scale_factor=self.image_size / 32, mode=mode),
            nn.Upsample(scale_factor=self.image_size / 32, mode=mode),
            nn.Upsample(scale_factor=self.image_size / 64, mode=mode),
            nn.Upsample(scale_factor=self.image_size / 64, mode=mode),
            nn.Upsample(scale_factor=self.image_size / 128, mode=mode),
            nn.Upsample(scale_factor=self.image_size / 128, mode=mode),
            nn.Upsample(scale_factor=self.image_size / 256, mode=mode),
            nn.Upsample(scale_factor=self.image_size / 256, mode=mode)
        ]
        for upsampler in self.upsamplers:
            upsampler.cuda()

    def load_json_data(self, json_data: Union[dict, list]):
        self.json_data = json_data

    def load_tensors(self, keys: List[str]):
        print("Loading DatasetGAN tensors...", end=" ")
        tensors = np.load(self.tensor_path, mmap_mode="r", allow_pickle=True)
        if "activations" in keys:
            self.activations = tensors["activations"]
        if "latent_codes" in keys:
            self.init_vectors = tensors["latent_codes"]
        del tensors
        print("Finished!")

    def get_feature_vector_length(self) -> int:
        return self.feature_vector_length

    def __len__(self):
        if self.random_sampling:
            sample_counts = [len(bucket) for bucket in self.sampling_buckets]
            return sum(sample_counts)
        else:
            return self.pixel_labels.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.random_sampling:
            rng = default_rng()
            indicator = rng.choice(len(self.class_probabilities), p=self.class_probabilities)

            bucket = self.sampling_buckets[indicator]
            indices = rng.choice(bucket)
            indices = tuple(indices)
        else:
            indices = numpy.unravel_index(index, self.pixel_labels.shape)

        activations = self.pixel_activations[indices]
        label = self.pixel_labels[indices]

        return {
            "activations": activations,
            "label": label
        }

    def create_sampling_buckets(self):
        for i in range(len(self.class_probabilities)):
            samples = np.argwhere(self.pixel_labels == i)
            self.sampling_buckets.append(samples)

    def get_images_for_plot(self, num_desired_images: int = 16) -> Tuple[List[torch.Tensor], List[torch.Tensor],
                                                                         List[torch.Tensor]]:
        images = []
        pixel_activations = []
        labels = []
        try:
            for i in range(num_desired_images):
                image = self.loader(self.image_paths[i])
                images.append(torchvision.transforms.ToTensor()(image))
                labels.append(torch.Tensor(self.pixel_labels[i]))
                pixel_activations.append(torch.Tensor(self.pixel_activations[i]))
        except IndexError:
            print(f"There are not enough samples in the dataset to match the number of desired images for the "
                  f"ImagePlotter. Using only {i} images")
        return images, pixel_activations, labels
