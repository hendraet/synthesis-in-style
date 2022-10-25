import argparse
import json
import math
import os
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Tuple

import numpy
from PIL import ImageColor
from matplotlib import pyplot as plt
from tqdm import tqdm

from segmentation.evaluation.evaluate_metrics import get_calculated_score_key_filters, extract_score_name
from segmentation.evaluation.evaluation_utils import get_tabular_results, preprocess_results, \
    group_results_by_hyperparam_values, add_mean_iou

if os.environ.get('REMOTE_PYCHARM_DEBUG_SESSION', False):
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=int(os.environ.get('REMOTE_PYCHARM_DEBUG_PORT')),
                            stdoutToServer=True, stderrToServer=True, suspend=False)


def set_box_color(boxplot: dict, color: Tuple):
    plt.setp(boxplot['medians'], color=color)


def plot_average_results(results: dict, class_names: list, results_color_palette: list,
                         figure_out_dir: Path, score_name: str):
    """
    For each hyperparameter boxplots are created that show the boxplot of given score name for each configuration.
    The plots are:
        - weighted average
        - class-wise scores side by side (interval [0., 1.])
        - separate boxplot for each class (zoomed in)
    """
    tabular_results, score_class_names, hyperparam_names = get_tabular_results(results, score_name)

    num_classes = len(score_class_names) - 1
    num_hyperparams = len(hyperparam_names)
    num_rows = 2 + num_classes
    fig_scale_factor = 6.
    fig, axs = plt.subplots(ncols=num_hyperparams, nrows=num_rows, figsize=(num_hyperparams * fig_scale_factor,
                                                                            num_rows * fig_scale_factor))
    for hyperparam_id in range(num_hyperparams):
        hyperparam_values = [str(v) for v in tabular_results[:, hyperparam_id]]

        # Plot weighted average dice scores as box plot
        weighted_average_index = len(hyperparam_names) + score_class_names.index('weighted_avg')
        weighted_average_score = tabular_results[:, weighted_average_index]
        grouped_results, labels = group_results_by_hyperparam_values(hyperparam_values, weighted_average_score)
        ax = axs[0, hyperparam_id]
        boxplot = ax.boxplot(grouped_results, labels=labels)
        set_box_color(boxplot, results_color_palette[0])
        ax.title.set_text(f'{hyperparam_names[hyperparam_id]} - weighted average')

        # Plot dice score for classes as box plot
        acc_ax = axs[1, hyperparam_id]
        for class_id in range(0, num_classes):
            result_column_id = num_hyperparams + class_id + 1
            result_column = tabular_results[:, result_column_id]
            grouped_results, labels = group_results_by_hyperparam_values(hyperparam_values, result_column)

            # accumulated boxplot
            shift = -0.8 * num_classes + class_id * 0.8
            plot_positions = numpy.array(range(len(grouped_results))) * num_classes + shift
            boxplot = acc_ax.boxplot(grouped_results, positions=plot_positions, sym='', widths=0.6, labels=labels)
            set_box_color(boxplot, results_color_palette[class_id + 1])
            acc_ax.title.set_text(f'{hyperparam_names[hyperparam_id]} - all classes')

            # single boxplot
            single_ax = axs[2 + class_id, hyperparam_id]
            boxplot = single_ax.boxplot(grouped_results, labels=labels)
            set_box_color(boxplot, results_color_palette[class_id + 1])
            single_ax.title.set_text(f'{hyperparam_names[hyperparam_id]} - {class_names[class_id]}')

    plt.setp(axs[1, :], ylim=(-0.05, 1.05))
    fig.suptitle(f"{results['general_config']['model_config']['network']} - {extract_score_name(score_name)}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure_out_path = figure_out_dir / f'average_results_{extract_score_name(score_name)}.png'
    fig.savefig(figure_out_path)
    fig.clf()


def plot_results_per_image(results: dict, figure_out_dir: Path, score_name: str):
    """
    For each image the scores of all runs are accumulated and a boxplot for each class is created
    """
    runs_per_image = defaultdict(list)
    for run in results['runs']:
        for k, v, in run[f'detailed_{score_name}_scores'].items():
            runs_per_image[k].append(v)

    runs_per_image = OrderedDict(sorted(runs_per_image.items()))

    num_cols = math.ceil(math.sqrt(len(runs_per_image)))
    num_rows = math.ceil(len(runs_per_image) / num_cols)
    fig, axs = plt.subplots(ncols=num_cols, nrows=num_rows, figsize=(num_cols * 6., num_rows * 6.))
    plt.setp(axs, ylim=(-0.05, 1.05))

    for image_id, (image_name, runs) in enumerate(runs_per_image.items()):
        box_plot_ready_results = list(zip(*[list((score_metrics['score'] for score_metrics in r.values())) for r in runs]))
        labels = list(runs[0].keys())

        row = image_id // num_cols
        col = image_id % num_cols
        ax = axs[row, col]
        ax.boxplot(box_plot_ready_results, labels=labels)
        ax.title.set_text(image_name)

    fig.suptitle(f"{results['general_config']['model_config']['network']} - {score_name}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(figure_out_dir / f'average_results_per_image_{score_name}.png')
    fig.clf()


def plot_class_wise_results(results: dict, figure_out_dir: Path, score_name: str):
    """
    Accumulates the dice scores of all runs for each class and creates a boxplot
    """
    class_wise_scores = defaultdict(list)
    for run in results['runs']:
        for scores in run[f'detailed_{score_name}_scores'].values():
            for cls, score_metrics in scores.items():
                class_wise_scores[cls].append(score_metrics['score'])

    fig = plt.figure(figsize=(6, 6))
    plt.boxplot(tuple(class_wise_scores.values()), labels=tuple(class_wise_scores.keys()))
    fig.suptitle(f"{results['general_config']['model_config']['network']} - {score_name}")
    fig.savefig(figure_out_dir / f'class_wise_results_{score_name}.png')
    fig.clf()


def main(args: argparse.Namespace):
    with open(args.results_path, 'r') as results_json:
        results = json.load(results_json)

    class_names = ['weighted_text_avg'] + list(results['general_config']['class_to_color_map'].keys())
    class_colors = [ImageColor.getrgb(color) for color in results['general_config']['class_to_color_map'].values()]
    # Add an additional color for weighted average
    results_color_palette = [ImageColor.getrgb('#006400'), ImageColor.getrgb('#640064')] + class_colors

    if args.calculate_mean_iou:
        add_mean_iou(results)

    results_color_palette = [tuple([cv / 255 for cv in color]) for color in results_color_palette]

    preprocess_results(results)

    args.figure_out_dir.mkdir(exist_ok=True, parents=True)

    score_key_filters = get_calculated_score_key_filters(results, score_key='average')
    for score_key_filter in tqdm(score_key_filters):
        plot_average_results(results, class_names, results_color_palette, args.figure_out_dir, score_key_filter)
        plot_results_per_image(results, args.figure_out_dir, extract_score_name(score_key_filter))
        plot_class_wise_results(results, args.figure_out_dir, extract_score_name(score_key_filter))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plots various metrics for an the results of a segmentation model')
    parser.add_argument('results_path', type=Path, help='Path to the results.json of the evaluation.')
    parser.add_argument('figure_out_dir', type=Path,
                        help='Path to the directory where the resulting plot should be saved to.')
    parser.add_argument('-c', '--calculate-mean-iou', action='store_true', default=False,
                        help='Replace the weighted average IoU with the unweighted mean IoU.')
    parsed_args = parser.parse_args()
    main(parsed_args)
