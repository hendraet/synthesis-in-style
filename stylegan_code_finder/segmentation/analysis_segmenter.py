import json
import math
from pathlib import Path
from typing import List, Union, Tuple, Iterator, NoReturn, Optional

import torch
from PIL.Image import Image as ImageClass
from torchvision import transforms
from tqdm import tqdm

from networks.base_segmenter import BaseSegmenter
from training_builder.train_builder_selection import get_train_builder_class
from utils.config import load_config
from utils.segmentation_utils import BBox
from visualization.utils import network_output_to_color_image

Color = Tuple[int, int, int]


class AnalysisSegmenter:

    def __init__(self, model_checkpoint: str, device: str, class_to_color_map: Union[str, Path],
                 original_config_path: Optional[Path] = None, max_image_size: int = None, print_progress: bool = True,
                 patch_overlap: int = 0, patch_overlap_factor: float = 0.0,
                 show_confidence_in_segmentation: bool = False):
        self.config = load_config(model_checkpoint, original_config_path)
        self.config['fine_tune'] = model_checkpoint
        self.class_to_color_map = self.load_color_map(class_to_color_map)
        self.device = device
        self.patch_size = int(self.config['image_size'])
        self.print_progress = print_progress
        self.max_image_size = max_image_size
        self.show_confidence_in_segmentation = show_confidence_in_segmentation
        self.network = self.load_network()

        self.set_patch_overlap(patch_overlap, patch_overlap_factor)

    def set_patch_overlap(self, patch_overlap: int, patch_overlap_factor: float) -> NoReturn:
        assert patch_overlap == 0 or patch_overlap_factor == 0.0, "Only one of 'patch_overlap' and " \
                                                                "'patch_overlap_factor' should be specified "
        if patch_overlap != 0:
            assert 0 < patch_overlap < self.patch_size, f"The value of 'patch_overlap' should be in the following " \
                                                        f"range: 0 < patch_overlap < patch_size ({self.patch_size} px) "
            self.patch_overlap = patch_overlap
        elif patch_overlap_factor != 0.0:
            assert 0.0 < patch_overlap_factor < 1.0, f"The value of 'patch_overlap_factor' should be in the " \
                                                     f"following range: 0.0 < patch_overlap_factor < 1.0 "
            self.patch_overlap = math.ceil(patch_overlap_factor * self.patch_size)
        else:
            self.patch_overlap = None

    def set_hyperparams(self, hyperparam_config: dict) -> NoReturn:
        if "patch_overlap" in hyperparam_config:
            self.set_patch_overlap(*hyperparam_config["patch_overlap"])
        if "min_confidence" in hyperparam_config:
            self.network.min_confidence = hyperparam_config["min_confidence"]
        if "min_contour_area" in hyperparam_config:
            self.network.min_contour_area = hyperparam_config["min_contour_area"]

    def progress_bar(self, *args, **kwargs):
        if self.print_progress:
            return tqdm(*args, **kwargs)
        else:
            return tuple(*args)

    def load_color_map(self, color_map_file: Union[str, Path]) -> dict:
        color_map_file = Path(color_map_file)
        with color_map_file.open() as f:
            color_map = json.load(f)
        return color_map

    def load_network(self) -> BaseSegmenter:
        # patch to support old config files where DocUFCN was the only available model
        if self.config['network'] == 'base':
            self.config['network'] = 'DocUFCN'
        train_builder_class = get_train_builder_class(self.config)
        training_builder = train_builder_class(self.config)
        segmentation_network = training_builder.get_network()
        segmentation_network.eval()

        return segmentation_network

    def calculate_bboxes_for_patches(self, image_width: int, image_height: int) -> Tuple[BBox]:
        patches = []
        if self.patch_overlap is not None:
            current_x, current_y = (0, 0)
            while current_y < image_height:
                while current_x < image_width:
                    image_box = BBox(current_x, current_y, current_x + self.patch_size,
                                     current_y + self.patch_size)
                    patches.append(image_box)
                    current_x += self.patch_size - self.patch_overlap
                current_x = 0
                current_y += self.patch_size - self.patch_overlap
        else:
            # automatic overlap calculation
            windows_in_width = math.ceil(image_width / self.patch_size)
            total_width_overlap = windows_in_width * self.patch_size - image_width
            windows_in_height = math.ceil(image_height / self.patch_size)
            total_height_overlap = windows_in_height * self.patch_size - image_height

            width_overlap_per_patch = total_width_overlap // windows_in_width
            height_overlap_per_patch = total_height_overlap // windows_in_height

            for y_idx in range(windows_in_height):
                start_y = int(y_idx * (self.patch_size - height_overlap_per_patch))
                for x_idx in range(windows_in_width):
                    start_x = int(x_idx * (self.patch_size - width_overlap_per_patch))
                    image_box = BBox(start_x, start_y, start_x + self.patch_size, start_y + self.patch_size)
                    patches.append(image_box)

        return tuple(patches)

    def crop_patches(self, input_image: ImageClass) -> Iterator[dict]:
        bboxes_for_patches = self.calculate_bboxes_for_patches(*input_image.size)
        for bbox in bboxes_for_patches:
            yield {
                 "image": input_image.crop(bbox),
                 "bbox": bbox
            }

    def predict_patches(self, patches: Iterator[dict]) -> [dict]:
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        transform_list = transforms.Compose(transform_list)

        predicted_patches = []
        for patch in self.progress_bar(patches, desc="Predicting patches...", leave=False):
            image = transform_list(patch["image"])
            batch = torch.unsqueeze(image, 0).to(self.device)
            with torch.no_grad():
                prediction = self.network.predict(batch)

            predicted_patches.append({
                "prediction": torch.squeeze(torch.detach(prediction), dim=0),
                "bbox": patch["bbox"]
            })

        return predicted_patches

    def assemble_predictions(self, patches: List[dict], output_size: Tuple) -> torch.Tensor:
        # dimensions are height, width, class for easier access
        num_classes = self.network.num_classes
        max_width = output_size[0]
        max_height = output_size[1]
        assembled_predictions = torch.full((max_height, max_width, num_classes), float("-inf"), device=self.device)

        for patch in self.progress_bar(patches, desc="Merging patches...", leave=False):
            reordered_patch = patch["prediction"].permute(1, 2, 0)
            x_start, y_start, x_end, y_end = patch["bbox"]
            x_end = min(x_end, max_width)
            y_end = min(y_end, max_height)
            window_height = y_end - y_start
            window_width = x_end - x_start

            assembled_window = assembled_predictions[y_start:y_end, x_start:x_end, :]
            patch_without_padding = reordered_patch[:window_height, :window_width, :]
            max_values = torch.maximum(assembled_window, patch_without_padding)
            assembled_predictions[y_start:y_end, x_start:x_end, :] = max_values

        return assembled_predictions.permute(2, 0, 1)  # permute so that the shape matches the original network output

    def convert_image_to_correct_color_space(self, image: ImageClass) -> ImageClass:
        if self.network.num_input_channels == 3:
            image = image.convert('RGB')
        elif self.network.num_input_channels == 1:
            image = image.convert('L')
        else:
            raise ValueError("Can not convert input image to desired format, Network desires inputs with "
                             f"{self.network.num_input_channels} channels.")
        return image

    def segment_image(self, image: ImageClass) -> torch.Tensor:
        image = self.convert_image_to_correct_color_space(image)

        if self.max_image_size > 0 and any(side > self.max_image_size for side in image.size):
            image.thumbnail((self.max_image_size, self.max_image_size))

        patches = self.crop_patches(image)

        with torch.no_grad():
            predicted_patches = self.predict_patches(patches)
            assembled_prediction = self.assemble_predictions(predicted_patches, image.size)

        return assembled_prediction

    def prediction_to_color_image(self, assembled_prediction: torch.tensor) -> ImageClass:
        full_img_tensor = network_output_to_color_image(torch.unsqueeze(assembled_prediction, dim=0),
                                                        self.class_to_color_map,
                                                        show_confidence_in_segmentation=self.show_confidence_in_segmentation)
        segmented_image = transforms.ToPILImage()(torch.squeeze(full_img_tensor, 0))
        return segmented_image


class VotingAssemblySegmenter(AnalysisSegmenter):

    def assemble_predictions(self, patches: List[dict], output_size: Tuple) -> torch.Tensor:
        # dimensions are height, width, class for easier access
        num_classes = self.network.num_classes
        max_width = output_size[0]
        max_height = output_size[1]
        summed_confidences = torch.zeros((num_classes, max_height, max_width), device=self.device)

        for patch in self.progress_bar(patches, desc="Merging patches...", leave=False):
            x_start, y_start, x_end, y_end = patch["bbox"]
            x_start = max(x_start, 0)
            y_start = max(y_start, 0)
            x_end = min(x_end, max_width)
            y_end = min(y_end, max_height)
            window_height = y_end - y_start
            window_width = x_end - x_start

            summed_confidences[:, y_start:y_end, x_start:x_end] += patch["prediction"][:, :window_height, :window_width]

        # Normalize votes to range [0, 1]
        predicted_class_confidences = summed_confidences / torch.unsqueeze(summed_confidences.sum(dim=0), dim=0)
        # If all confidences are 0, division will lead to nan. This is likely due to low confidences that were removed
        # during postprocessing
        predicted_class_confidences = torch.nan_to_num(predicted_class_confidences)
        return predicted_class_confidences
