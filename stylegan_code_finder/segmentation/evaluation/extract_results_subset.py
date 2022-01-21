import argparse
import json
from pathlib import Path
from typing import List

import torch

from segmentation.evaluation.evaluation_utils import get_calculated_score_key_filters, extract_score_name
from segmentation.evaluation.segmentation_metric_calculation import calculate_metric


def remove_not_in_subset(results: dict, subset: List[str]):
    metrics = ["confusion_matrices"] + get_calculated_score_key_filters(results, "detailed")

    for i in range(len(results["runs"])):
        for metric in metrics:
            keys = list(results["runs"][i][metric].keys())
            for key in keys:
                if key not in subset:
                    results["runs"][i][metric].pop(key, None)


def recompute_metrics(results: dict):
    keys = get_calculated_score_key_filters(results, "average")
    class_names = list(results["runs"][0][keys[0]].keys())
    class_names = [i for i in class_names if "weighted" not in i]

    for i in range(len(results["runs"])):
        subset_confusion_matrix = compute_confusion_matrix(results["runs"][i], len(class_names))

        for key in keys:
            metric = extract_score_name(key)
            new_scores = calculate_metric(subset_confusion_matrix, class_names, metric)
            results["runs"][i][key] = new_scores


def compute_confusion_matrix(run: dict, num_classes: int) -> torch.Tensor:
    sample_matrices = []

    for key in run["confusion_matrices"].keys():
        sample_matrix = torch.Tensor(run["confusion_matrices"][key])
        sample_matrix = sample_matrix.reshape((num_classes, num_classes))
        sample_matrices.append(sample_matrix)

    subset_matrix = torch.stack(sample_matrices, dim=0).sum(dim=0)

    return subset_matrix


def main(args: argparse.Namespace):
    with open(args.results_path, "r") as results_json:
        results = json.load(results_json)

    with open(args.subset_path, "r") as subset_file:
        lines = subset_file.readlines()
        subset = [line.rsplit(".", 1)[0] for line in lines]

    remove_not_in_subset(results, subset)
    recompute_metrics(results)

    with open(args.subset_results_path, "w") as out_json:
        json.dump(results, out_json, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Computes the metrics of a segmenter evaluation but only for a given subset of "
                                     "the evaluation subset")
    parser.add_argument("results_path", type=Path, help="Path to the results.json of the evaluation.")
    parser.add_argument("subset_path", type=Path,
                        help="Path to a text file containing the subset of images to extract.")
    parser.add_argument("subset_results_path", type=Path,
                        help="Path to the file where the extracted subset should be saved to.")

    parsed_args = parser.parse_args()
    main(parsed_args)
