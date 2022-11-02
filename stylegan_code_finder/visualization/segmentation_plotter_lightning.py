import json
from pathlib import Path
from typing import List
import os
import numpy
import torch
from PIL import ImageColor
from PIL import Image
import torchvision
from pytorch_training.images.utils import make_image
from utils.data_loading import fill_plot_images


class SegmentationPlotter:
    def __init__(self, plot_data_loader, config):
        assert config['class_to_color_map'] is not None, "Class to Color map must be supplied to SegmentationPlotter!"
        with Path(config['class_to_color_map']).open() as f:
            self.class_to_color_map = json.load(f)
        plot_images = fill_plot_images(plot_data_loader, num_desired_images=config['display_size'])
        self.label_images = self.label_images_to_color_images(torch.stack(plot_images['segmented']).cuda())
        self.input_images = torch.stack(plot_images['images']).cuda()
        self.counter = 0
        self.image_dir = os.path.join(config['log_dir'], 'images')
        Path(self.image_dir).mkdir(parents=True, exist_ok=True)


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

    def get_predictions(self, network) -> List[torch.Tensor]:
        predictions = [self.input_images, self.label_images]
        with torch.no_grad():
            predicted_classes = network.predict_classes(self.input_images)
        network_output = self.label_images_to_color_images(predicted_classes)
        predictions.append(network_output)
        return predictions

    def run(self, network):
        torch.cuda.empty_cache()
        try:
            network.eval()
            with torch.no_grad():
                predictions = self.get_predictions(network)
        finally:
            network.train()

        display_images = torch.cat(predictions, dim=0)

        image_grid = torchvision.utils.make_grid(display_images, nrow=self.input_images.shape[0])

        dest_file_name = os.path.join(self.image_dir, f"{self.counter:08d}.png")
        self.counter += 1
        dest_image = make_image(image_grid)
        Image.fromarray(dest_image).save(dest_file_name)

        del display_images
        torch.cuda.empty_cache()
        return dest_file_name
