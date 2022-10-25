import cv2
import warnings

import numpy
from PIL import ImageColor, Image
from PIL.ImageDraw import ImageDraw
from tqdm import tqdm
from typing import Dict, Union, Tuple, List, NoReturn

from pathlib import Path

import argparse
import torch

from PIL.Image import Image as ImageClass
from utils.image_utils import opencv_image_to_pil, pil_image_to_opencv
from utils.segmentation_utils import BBox, find_class_contours

Color = Tuple[int, int, int]


def save_overlayed_segmentation_image(segmented_image: ImageClass, original_image: ImageClass, class_to_color_map: dict,
                                      image_save_dir: Path, image_prefix: str):
    segmented_image_array = numpy.array(segmented_image)
    original_image_array = numpy.array(original_image)
    background_color = ImageColor.getrgb(class_to_color_map['background'])

    colored_pixel_indices = numpy.where((segmented_image_array != background_color))[:-1]
    original_image_array[colored_pixel_indices] = segmented_image_array[colored_pixel_indices]

    overlayed_image = Image.fromarray(original_image_array)
    overlayed_image.save(image_save_dir / f"{image_prefix}_overlayed.png")


def get_bounding_boxes(prediction: torch.Tensor, filter_classes: Tuple = ()) -> Dict[int, List[BBox]]:
    class_contours = find_class_contours(prediction, background_class_id=0, filter_classes=filter_classes)

    bbox_dict = {}
    for class_id, contours in class_contours.items():
        bbox_dict[class_id] = []
        for contour in contours:
            bbox = BBox.from_opencv_bounding_rect(*cv2.boundingRect(contour))
            bbox_dict[class_id].append(bbox)

    return bbox_dict


def draw_bounding_boxes(image: ImageClass, bboxes: Union[Tuple[BBox], List[BBox]], outline_color: Color = (0, 255, 0),
                        stroke_width: int = 3) -> NoReturn:
    d = ImageDraw(image)
    for bbox in bboxes:
        d.rectangle(bbox, outline=outline_color, width=stroke_width)


def draw_segmentation(original_image: ImageClass, assembled_predictions: torch.Tensor, original_segmented_image: Image,
                      bboxes_for_patches: Union[Tuple[BBox], None] = None, filter_classes: Tuple[int, ...] = (),
                      return_bboxes: bool = False) -> Union[
    Tuple[ImageClass, ImageClass], Tuple[ImageClass, ImageClass, Dict[int, List[BBox]]]]:
    if original_image.size != original_segmented_image.size:
        warnings.warn("Sizes of original_image and original_segmented_image do not match. It could be that there is "
                      "something wrong with the preprocessing of these images.")
    bbox_dict = get_bounding_boxes(assembled_predictions, filter_classes=filter_classes)
    bboxes = tuple([bbox for bboxes in list(bbox_dict.values()) for bbox in bboxes])

    segmented_image = original_segmented_image.copy()
    draw_bounding_boxes(segmented_image, bboxes)

    image = original_image.copy()
    draw_bounding_boxes(image, bboxes)

    if bboxes_for_patches is not None:
        draw_bounding_boxes(image, bboxes_for_patches, outline_color=(255, 0, 0), stroke_width=1)
        draw_bounding_boxes(segmented_image, bboxes_for_patches, outline_color=(255, 0, 0), stroke_width=1)

    if return_bboxes:
        return image, segmented_image, bbox_dict
    else:
        return image, segmented_image


def save_bbox_to_image(bbox: BBox, original_image: Union[ImageClass, numpy.ndarray], filename: Path) -> NoReturn:
    if isinstance(original_image, numpy.ndarray):
        original_image = opencv_image_to_pil(original_image)
    cropped_area = original_image.crop(bbox)
    cropped_area.save(filename)


def extract_and_save_bounding_boxes(image: ImageClass, assembled_predictions: torch.Tensor, output_dir: Path,
                                    image_prefix: str, filter_classes: Tuple[int, ...] = ()) -> NoReturn:
    bbox_dir = output_dir / "bboxes"
    bbox_dir.mkdir(exist_ok=True)
    bbox_dict = get_bounding_boxes(assembled_predictions, filter_classes=filter_classes)
    for class_id, bboxes in bbox_dict.items():
        for i, bbox in enumerate(tqdm(bboxes, desc="Cropping bboxes...", leave=False)):
            filename = bbox_dir / f"{image_prefix}_class_{class_id}_bbox_{i}.png"
            save_bbox_to_image(bbox, image, filename)


def save_contour_to_image(contour: numpy.ndarray, original_image: Union[ImageClass, numpy.ndarray], filename: Path,
                          background_color: Color = (255, 255, 255)) -> NoReturn:
    if isinstance(original_image, ImageClass):
        original_image = pil_image_to_opencv(original_image)

    mask = numpy.zeros_like(original_image)
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)

    full_output_image = numpy.zeros_like(original_image)
    output_channels = []
    for i, (mask_channel, original_channel) in enumerate(zip(cv2.split(mask), cv2.split(original_image))):
        output_channel = numpy.where(mask_channel == 255, original_channel, background_color[i])
        output_channels.append(output_channel)
    cv2.merge(output_channels, full_output_image)

    x, y, w, h = cv2.boundingRect(contour)
    if len(full_output_image.shape) == 3:
        output_image = full_output_image[y:y + h, x:x + w, :]
    elif len(full_output_image.shape) == 2:
        output_image = full_output_image[y:y + h, x:x + w]
    else:
        raise NotImplementedError

    cv2.imwrite(str(filename), output_image)


def extract_and_save_contours(image: ImageClass, assembled_predictions: torch.Tensor, output_dir: Path,
                              image_prefix: str) -> NoReturn:
    contour_dir = output_dir / "contours"
    contour_dir.mkdir(exist_ok=True)
    class_contours = find_class_contours(assembled_predictions, background_class_id=0)
    for class_id, contours in class_contours.items():
        for i, contour in enumerate(tqdm(contours, desc=f"Extracting contours for class {class_id}...", leave=False)):
            filename = contour_dir / f"{image_prefix}_class_{class_id}_contour_{i}.png"
            save_contour_to_image(contour, image, filename)


def visualize_segmentation(assembled_prediction: torch.Tensor, image: ImageClass, original_image: ImageClass,
                           segmenter: 'AnalysisSegmenter', args: argparse.Namespace, class_to_color_map: Dict,
                           image_prefix: str) -> NoReturn:
    image_save_dir = args.output_dir / "images"
    image_save_dir.mkdir(exist_ok=True, parents=True)

    # segmented image is an image that only displays the color-coded, predicted class for each pixel
    segmented_image = segmenter.prediction_to_color_image(assembled_prediction)
    segmented_image.save(image_save_dir / f"{image_prefix}_segmented_no_bboxes.png")

    if args.overlay_segmentation:
        save_overlayed_segmentation_image(segmented_image, original_image, class_to_color_map, image_save_dir,
                                          image_prefix)

    if args.extract_bboxes:
        bboxes_for_patches = segmenter.calculate_bboxes_for_patches(*image.size) if args.draw_patches else None
        image_w_bboxes, segmented_image_w_bboxes = draw_segmentation(
            original_image,
            assembled_prediction,
            segmented_image,
            bboxes_for_patches=bboxes_for_patches
        )

        image_w_bboxes.save(image_save_dir / f"{image_prefix}_bboxes.png")
        if args.draw_bboxes_on_segmentation:
            segmented_image_w_bboxes.save(image_save_dir / f"{image_prefix}_segmented.png")

    if args.save_bboxes:
        extract_and_save_bounding_boxes(image, assembled_prediction, image_save_dir, image_prefix)
    if args.save_contours:
        extract_and_save_contours(image, assembled_prediction, image_save_dir, image_prefix)
