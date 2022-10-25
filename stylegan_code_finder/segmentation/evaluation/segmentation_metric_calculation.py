from collections import defaultdict
from pathlib import Path

from typing import Dict, List

import argparse
import numpy
import torch
from PIL import Image
from torch.nn.functional import one_hot

from utils.segmentation_utils import segmentation_image_to_class_image


def get_ground_truth_class_map_for_image(image_path: Path, ground_truth_dir: Path, class_to_color_map: dict,
                                         num_classes, background_class_name="background") -> torch.Tensor:
    gt_path = ground_truth_dir / f"{image_path.stem}_gt.png"
    assert gt_path.exists(), f"The following ground truth image does not exist: {gt_path}. Is it a png?"
    gt_image = numpy.asarray(Image.open(gt_path).convert("RGB"))
    gt_classes = segmentation_image_to_class_image(gt_image, background_class_name, class_to_color_map)

    return torch.as_tensor(gt_classes, dtype=torch.int64)


def calculate_dice_score(confusion_matrix: torch.Tensor, class_idx: int) -> float:
    true_positives = confusion_matrix[class_idx, class_idx]
    predicted_positives = confusion_matrix[:, class_idx].sum()
    actual_positives = confusion_matrix[class_idx, :].sum()

    dice_score = 2 * true_positives / (predicted_positives + actual_positives)
    # If the class isn't present in the gt nor in the prediction, then the prediction was correct
    return torch.nan_to_num(dice_score, nan=1.0).item()


def calculate_iou(confusion_matrix: torch.Tensor, class_idx: int) -> float:
    true_positives = confusion_matrix[class_idx, class_idx]
    predicted_positives = confusion_matrix[:, class_idx].sum()
    actual_positives = confusion_matrix[class_idx, :].sum()

    iou = true_positives / (predicted_positives + actual_positives - true_positives)
    # If the class isn't present in the gt nor in the prediction, then the prediction was correct
    return torch.nan_to_num(iou, nan=1.0).item()


def calculate_precision(confusion_matrix: torch.Tensor, class_idx: int) -> float:
    true_positives = confusion_matrix[class_idx, class_idx]
    predicted_positives = confusion_matrix[:, class_idx].sum()

    precision = true_positives / predicted_positives
    # If the class isn't present in the gt nor in the prediction, then the prediction was correct
    return torch.nan_to_num(precision, nan=1.0).item()


def calculate_recall(confusion_matrix: torch.Tensor, class_idx: int) -> float:
    true_positives = confusion_matrix[class_idx, class_idx]
    actual_positives = confusion_matrix[class_idx, :].sum()

    recall = true_positives / actual_positives
    # If the class isn't present in the gt nor in the prediction, then the prediction was correct
    return torch.nan_to_num(recall, nan=1.0).item()


IMPLEMENTED_METRICS = {
    "dice": calculate_dice_score,
    "iou": calculate_iou,
    "precision": calculate_precision,
    "recall": calculate_recall
}


def calculate_confusion_matrix(assembled_prediction: torch.Tensor, image_path: Path, args: argparse.Namespace,
                               class_to_color_map: dict, num_classes: int) -> torch.Tensor:
    confusion_matrix = torch.zeros((num_classes, num_classes))

    # If two confidences are equal, argmax will choose the index of the first class. Thus, the model is, theoretically,
    # slightly biased in favor of classes with lower ids.
    # However, this can only occur when using the VotingAssemblySegmenter and since it is float-based, it is highly
    # unlikely that the summed confidences for two classes are exactly the same.
    predicted_classes = torch.argmax(assembled_prediction, dim=0)

    ground_truth_classes = get_ground_truth_class_map_for_image(image_path, args.ground_truth_dir,
                                                                  class_to_color_map, num_classes)
    assert predicted_classes.shape == ground_truth_classes.shape, 'Shapes of prediction and ground ' \
                                                                      'truth do not match'

    ground_truth_classes = ground_truth_classes.cuda()
    predicted_classes = predicted_classes.cuda()

    for i in range(num_classes):
        for j in range(num_classes):
            sample_mask = torch.logical_and((ground_truth_classes == i), (predicted_classes == j))
            confusion_matrix[i, j] = sample_mask.double().sum().cpu().item()

    return confusion_matrix


def calculate_metric(confusion_matrix: torch.Tensor, class_names: List[str],
                     metric: str = "dice") -> Dict[str, Dict[str, float]]:
    assert metric in IMPLEMENTED_METRICS.keys(), \
        f"Metric to calculate must be in {', '.join(m for m in IMPLEMENTED_METRICS.keys())}"

    scores = {
        "weighted_avg": {"score": 0.0},
        "weighted_text_avg": {"score": 0.0}
    }

    total_text_weight = 0.0

    for class_idx, name in enumerate(class_names):
        score = IMPLEMENTED_METRICS[metric](confusion_matrix, class_idx)
        weight = confusion_matrix[class_idx, :].sum().item() / confusion_matrix.sum().item()

        if "text" in name:
            total_text_weight += weight

        scores["weighted_avg"]["score"] += score * weight
        scores[class_names[class_idx]] = {"score": score, "weight": weight}

    for name in class_names:
        if "text" in name:
            score = scores[name]["score"]
            weight = scores[name]["weight"]

            if total_text_weight > 0:
                scores["weighted_text_avg"]["score"] += score * weight / total_text_weight
            else:
                scores["weighted_text_avg"]["score"] = 1.0

    return scores
