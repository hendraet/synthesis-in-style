import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import numpy
import pandas

from segmentation.evaluation.evaluation_utils import preprocess_results, get_tabular_results, \
    group_results_by_hyperparam_values, get_calculated_score_key_filters, add_mean_iou, \
    extract_score_name

if os.environ.get("REMOTE_PYCHARM_DEBUG_SESSION", False):
    import pydevd_pycharm

    pydevd_pycharm.settrace('localhost', port=int(os.environ.get("REMOTE_PYCHARM_DEBUG_PORT")),
                            stdoutToServer=True, stderrToServer=True, suspend=False)


def print_data_frame(data_frame: pandas.DataFrame, title: str = ""):
    if title != "":
        print(title)
    print(data_frame.to_markdown() + "\n")


def get_best_results(results: dict, score_key: str) -> pandas.DataFrame:
    best_hyperparam_configs = {}
    sorted_hyperparam_configs = defaultdict(list)
    for run in results["runs"]:
        for k, v in run[score_key].items():
            sorted_hyperparam_configs[k].append((v["score"], run["hyperparams"]))
            if v["score"] > best_hyperparam_configs.get(k, {score_key: 0., })[score_key]:
                # Merge the two dictionaries and store them in k
                best_hyperparam_configs[k] = {score_key: v["score"], **run["hyperparams"]}

    pandas_ready_dict = defaultdict(list)
    for class_name, metrics in best_hyperparam_configs.items():
        pandas_ready_dict["class"].append(class_name)
        for k, v in metrics.items():
            pandas_ready_dict[k].append(v)

    return pandas.DataFrame(data=pandas_ready_dict)


def get_best_median_configs(tabular_results: numpy.ndarray, hyperparam_names: tuple,
                            score_class_names: tuple, score_name: str) -> Dict[str, pandas.DataFrame]:
    best_median_hyperparam_configs = defaultdict(dict)
    num_hyperparams = len(hyperparam_names)
    for hyperparam_id, hyperparam_name in enumerate(hyperparam_names):
        hyperparam_values = [str(v) for v in tabular_results[:, hyperparam_id]]

        for class_id in range(len(score_class_names)):
            result_column_id = num_hyperparams + class_id
            result_column = tabular_results[:, result_column_id]

            grouped_results, labels = group_results_by_hyperparam_values(hyperparam_values, result_column)
            medians = [(l, statistics.median(group)) for l, group in zip(labels, grouped_results)]
            sorted_medians = sorted(medians, key=lambda x: x[1], reverse=True)
            best_median_hyperparam_configs[hyperparam_name][score_class_names[class_id]] = sorted_medians[0]

    best_median_configs = {}
    for hyperparam_name, metrics in best_median_hyperparam_configs.items():
        pandas_ready_dict = defaultdict(list)
        for k, v in metrics.items():
            pandas_ready_dict["class"].append(k)
            pandas_ready_dict["hyperarameter value"].append(v[0])
            pandas_ready_dict[f"best median {score_name} score"].append(v[1])
        best_median_configs[hyperparam_name] = pandas.DataFrame(data=pandas_ready_dict)

    return best_median_configs


def print_result_tables(results: dict):
    score_key_filters = get_calculated_score_key_filters(results, score_key="average")
    iou_exists = False
    for score_key_filter in score_key_filters:
        tabular_results, score_class_names, hyperparam_names = get_tabular_results(results, score_key_filter)

        print(f"# {results['general_config']['model_config']['network']} - {extract_score_name(score_key_filter)}\n")
        print("## Best Results\n")
        best_results = get_best_results(results, score_key_filter)
        print_data_frame(best_results)

        if score_key_filter == "average_iou_scores":
            iou_exists = True
            min_confidence = float(best_results.loc[best_results['class'] == "weighted_avg"]["min_confidence"])
            min_contour_area = int(best_results.loc[best_results['class'] == "weighted_avg"]["min_contour_area"])
            patch_overlap = float(best_results.loc[best_results['class'] == "weighted_avg"]["patch_overlap"])
            best_miou_params = f"min_confidence {min_confidence} min_contour_area {min_contour_area} patch_overlap {patch_overlap}"

        print("## Median Results for each Hyperparameter\n")
        best_median_configs = get_best_median_configs(tabular_results, hyperparam_names, score_class_names,
                                                      extract_score_name(score_key_filter))
        for hyperparam_name, df in best_median_configs.items():
            print_data_frame(df, title=f"### {hyperparam_name}")
    if iou_exists:
        print("# Config for best mIoU\n")
        results = get_result_for_given_config(best_miou_params.split(" "), results)
        df = get_dataframe_from_results(results)
        df = df.rename(columns={"iou_weighted_avg": "mIoU", "iou_weighted_text_avg": "mIoU_text_only"}).T
        print_data_frame(df.head(3))

        print("# All Metrics for best mIoU\n")
        print_data_frame(df.tail(len(df) - 3))


def are_configs_matching(hyperparam_config: dict, run_config: dict) -> bool:
    for k, v, in hyperparam_config.items():
        assert k in run_config, f"{k} is not present in the hyperparameter config of the results"
        type_converter = type(run_config[k])
        if not (run_config[k] == type_converter(v)):
            return False
    return True


def get_result_for_given_config(raw_config: List[str], results: dict) -> List[dict]:
    hyperparam_config = dict(zip(raw_config[::2], raw_config[1::2]))
    matching_results = []
    for run in results["runs"]:
        run_config = run["hyperparams"]
        if are_configs_matching(hyperparam_config, run_config):
            merged_results = run_config
            for score_key in filter(lambda x: "average" in x, run.keys()):
                score_name = extract_score_name(score_key)
                renamed_results = {f'{score_name}_{k}': v["score"] for k, v in run[score_key].items()}
                merged_results = {**merged_results, **renamed_results}
            matching_results.append(merged_results)
    return matching_results


def get_dataframe_from_results(matching_results: List[dict]) -> pandas.DataFrame:
    pandas_ready_dict = defaultdict(list)
    for result in matching_results:
        for k, v in result.items():
            pandas_ready_dict[k].append(v)
    return pandas.DataFrame(data=pandas_ready_dict)


def print_result_for_given_config(raw_config: List[str], results: dict):
    matching_results = get_result_for_given_config(raw_config, results)
    if len(matching_results) == 0:
        print("No matching config found.")
    else:
        df = get_dataframe_from_results(matching_results)
        print_data_frame(df)


def main(args: argparse.Namespace):
    with open(args.results_path, "r") as results_json:
        results = json.load(results_json)
    preprocess_results(results)

    if args.calculate_mean_iou:
        add_mean_iou(results)

    if args.print_tables:
        print_result_tables(results)
    elif args.get_result_for_config is not None:
        print_result_for_given_config(args.get_result_for_config, results)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Takes the evaluation results of a trained classifier and prints the formatted "
                                     "results")
    parser.add_argument("results_path", type=Path, help="Path to the results.json of the evaluation.")
    parser.add_argument("-c", "--calculate-mean-iou", action="store_true", default=False,
                        help="Replace the weighted average IoU with the unweighted mean IoU.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--print-tables", action="store_true", default=False,
                       help="Evaluate the results and print the best results as well as the median best results for "
                            "each hyperparameter.")
    group.add_argument("-r", "--get-result-for-config", nargs="+",
                       help="Get the average results for the run that fits the given config. Config should be a "
                            "space-separated list of hyperparameter names and their corresponding values, e.g. "
                            "min_confidence 0.0 min_contour_area 55 patch_overlap 0.0."
                            "If the config is only given partially, multiple results will be returned.")
    parsed_args = parser.parse_args()
    main(parsed_args)
