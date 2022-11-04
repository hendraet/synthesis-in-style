import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import os
import numpy
import torch
from PIL import ImageColor, Image
import torchvision
from pytorch_training.images.utils import make_image


class SegmentationPlotter:
    def __init__(self, config):
        assert config['class_to_color_map'] is not None, "Class to Color map must be supplied to SegmentationPlotter!"
        with Path(config['class_to_color_map']).open() as f:
            self.class_to_color_map = json.load(f)
        self.counter = 0
        self.image_dir = Path(config['log_dir'], 'images')
        self.image_dir.mkdir(parents=True, exist_ok=True)

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

    def get_predictions(self, network, input_images, label_images) -> List[torch.Tensor]:
        predictions = [input_images, label_images]
        predicted_classes = network.predict_classes(input_images)
        network_output = self.label_images_to_color_images(predicted_classes)
        predictions.append(network_output)
        return predictions

    def run(self, network, batch) -> str:
        torch.cuda.empty_cache()  # free up space in the GPU memory for the images
        plot_images = self.fill_one_batch(batch)
        input_images = torch.stack(plot_images['images'])
        label_images = self.label_images_to_color_images(torch.stack(plot_images['segmented']))
        predictions = self.get_predictions(network, input_images, label_images)

        display_images = torch.cat(predictions, dim=0)

        image_grid = torchvision.utils.make_grid(display_images, nrow=input_images.shape[0])

        dest_file_name = Path(self.image_dir, f"{self.counter:08d}.png")
        self.counter += 1
        dest_image = make_image(image_grid)
        output_image = Image.fromarray(dest_image)
        output_image.save(dest_file_name)

        del display_images
        torch.cuda.empty_cache()
        return output_image

    def fill_one_batch(self, batch) -> Dict[str, List[torch.Tensor]]:
        image_list = defaultdict(list)
        for image_key, images in batch.items():
            for image in images:
                image_list[image_key].append(image)
        return image_list
