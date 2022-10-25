from typing import Dict, List

import numpy
import torch
from PIL import ImageColor
from utils.segmentation_utils import Color


def linear_gradient(start_rgb: Color, finish_rgb: Color, n: int = 10) -> List[Color]:
    """
    returns a gradient list of (n) colors between two colors.
    Adapted from: https://gist.github.com/RoboDonut/83ec5909621a6037f275799032e97563
    """
    # Initialize a list of the output colors with the starting color
    RGB_list = [start_rgb]
    # Calculate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
            int(start_rgb[j] + (float(t) / (n - 1)) * (finish_rgb[j] - start_rgb[j]))
            for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(tuple(curr_vector))

    return RGB_list


def network_output_to_color_image(network_outputs: torch.Tensor, class_to_color_map: Dict,
                                  show_confidence_in_segmentation: bool = False) -> torch.Tensor:
    network_outputs = network_outputs.permute(0, 2, 3, 1)
    batch_size, height, width, num_predicted_classes = network_outputs.shape

    assert num_predicted_classes == len(class_to_color_map), "Number of predicted classes and expected classes does " \
                                                             "not match " f"{num_predicted_classes} vs " \
                                                             f"{len(class_to_color_map)}"
    segmentation_images = numpy.zeros((batch_size, height, width, 3), dtype=numpy.uint8)
    segmentation_images[:, :, :] = ImageColor.getrgb(class_to_color_map["background"])
    segmentation_images = torch.from_numpy(segmentation_images).to(network_outputs.device)

    if show_confidence_in_segmentation:
        # Create linear color gradients that will be used to reflect confidences
        steps = 100
        gradients = [linear_gradient((255, 255, 255), ImageColor.getrgb(color), steps) for color in
                     class_to_color_map.values()]

        # find all indices that might be something other than background, i.e. where one of the other classes has at
        # least one vote
        indices_not_background = torch.where(torch.sum(network_outputs[:, :, :, 1:], dim=-1) > 0)
        for indices in zip(*indices_not_background):
            votes = network_outputs[indices]
            class_idx = votes.argmax()
            color_strength = votes.max().item()
            final_color = gradients[class_idx][int(steps * color_strength) - 1]
            segmentation_images[indices] = torch.from_numpy(
                numpy.asarray(final_color, numpy.uint8)).to(network_outputs.device)
    else:
        predicted_classes = torch.max(network_outputs, dim=-1).indices.type(torch.float32)
        for class_id, (class_name, color) in enumerate(class_to_color_map.items()):
            if class_name == "background":
                continue
            colorization_mask = predicted_classes == class_id
            rgb_color = ImageColor.getrgb(color)
            segmentation_images[colorization_mask] = torch.from_numpy(
                numpy.asarray(rgb_color, numpy.uint8)).to(network_outputs.device)

    segmentation_images = (segmentation_images / 255)
    return segmentation_images.permute(0, 3, 1, 2)
