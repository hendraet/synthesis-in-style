import itertools
from typing import Tuple, Union, List

import numpy
import operator


def get_calculated_score_key_filters(results: dict, score_key: str = "average") -> List[str]:
    scores = [key for key in results["runs"][0].keys() if score_key in key]
    return scores


def extract_score_name(score_key_filter: str) -> str:
    score_key_parts = score_key_filter.split('_')
    assert len(score_key_parts) == 3, "score keys must consist of three parts delimited by '_' ('average_dice_score)"
    return score_key_parts[1]


def calculate_mean_iou(scores: dict) -> float:
    background = scores["background"]["score"]
    printed_text = scores["printed_text"]["score"]
    handwritten_text = scores["handwritten_text"]["score"]

    return (background + printed_text + handwritten_text) / 3.0


def add_mean_iou(results: dict):
    for i in range(len(results["runs"])):
        global_scores = results["runs"][i]["average_iou_scores"]
        global_miou = calculate_mean_iou(global_scores)
        results["runs"][i]["average_iou_scores"]["weighted_avg"]["score"] = global_miou

        for sample in results["runs"][i]["detailed_iou_scores"].keys():
            scores = results["runs"][i]["detailed_iou_scores"][sample]
            miou = calculate_mean_iou(scores)
            results["runs"][i]["detailed_iou_scores"][sample]["weighted_avg"]["score"] = miou


def preprocess_results(results: dict):
    for run in results["runs"]:
        if "patch_overlap" in run["hyperparams"]:
            assert run["hyperparams"]["patch_overlap"][0] == 0, "Code assumes that patch overlap is given as float."
            run["hyperparams"]["patch_overlap"] = run["hyperparams"]["patch_overlap"][1]


def group_results_by_hyperparam_values(hyperparam_values: list, results: numpy.ndarray) -> Tuple[list, list]:
    grouped_results = [list(el) for _, el in itertools.groupby(sorted(zip(hyperparam_values, results)),
                                                               operator.itemgetter(0))]
    formatted_results = [[el[1] for el in group] for group in grouped_results]
    labels = [group[0][0] for group in grouped_results]
    return formatted_results, labels


def get_tabular_results(results: dict, score_key: str) -> Union[numpy.ndarray, Tuple, Tuple]:
    """
    Arrange the evaluation results in a numpy table so that they can be accessed easily while plotting or
    calculating metrics
    """
    hyperparam_names = tuple(results["runs"][0]["hyperparams"].keys())
    score_class_names = tuple(results["runs"][0][score_key].keys())
    array_keys = hyperparam_names + score_class_names

    tabular_results = numpy.zeros((len(results["runs"]), len(array_keys)))
    for run_id, run in enumerate(results["runs"]):
        run_results = tuple(run["hyperparams"].values()) + tuple([val["score"] for val in run[score_key].values()])
        tabular_results[run_id] = run_results

    return tabular_results, score_class_names, hyperparam_names
