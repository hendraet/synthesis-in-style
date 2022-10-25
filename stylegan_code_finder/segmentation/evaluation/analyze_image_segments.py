import argparse
import itertools
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import NoReturn

import torch
from PIL import Image, UnidentifiedImageError
from PIL.Image import Image as ImageClass
from tqdm import tqdm

from pytorch_training.images import is_image
from segmentation.analysis_segmenter import VotingAssemblySegmenter
from segmentation.evaluation.segmentation_metric_calculation import calculate_confusion_matrix, calculate_metric
from segmentation.evaluation.segmentation_visualization import visualize_segmentation
from utils.image_utils import resize_image

if os.environ.get("REMOTE_PYCHARM_DEBUG_SESSION", False):
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=int(os.environ.get("REMOTE_PYCHARM_DEBUG_PORT")),
                            stdoutToServer=True, stderrToServer=True, suspend=False)


def parse_and_check_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze the given images using the specified segmentation model. Analysis may include the "
                    "visualization of the segmentation as well as the calculation of the dice scores. Detected "
                    "contours/bounding boxes can also be extracted and saved separately.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode control
    mode_group = parser.add_argument_group("Management of evaluation modes")
    mode_group.add_argument("-cds", "--calculate-dice-score", action="store_true", default=False,
                            help="If set, the dice score for all input images will be calculated. Requires the "
                                 "--ground-truth-dir argument to be set properly.")
    mode_group.add_argument("-cio", "--calculate-iou", action="store_true", default=False,
                            help="If set, the intersection over union for all input images will be calculated. "
                                 "Requires the --ground-truth-dir argument to be set properly.")
    mode_group.add_argument("-cpr", "--calculate-precision", action="store_true", default=False,
                            help="If set, the precision for all input images will be calculated. Requires the "
                                 "--ground-truth-dir argument to be set properly.")
    mode_group.add_argument("-cre", "--calculate-recall", action="store_true", default=False,
                            help="If set, the recall for all input images will be calculated. Requires the "
                                 "--ground-truth-dir argument to be set properly.")
    mode_group.add_argument("-vis", "--visualize-segmentation", action="store_true", default=False,
                            help="If set, the segmentation result will be visualized and images will be saved. See "
                                 "arguments below to manage the visualization output.")

    # Input and output file management
    file_group = parser.add_argument_group("File management")
    file_group.add_argument("image_dir", type=Path,
                            help="Path to a directory that contains the images that should be analyzed")
    file_group.add_argument("-f", "--config-file", default="config.json", type=Path,
                            help="Path to the JSON file that contains the segmenter configuration")
    file_group.add_argument("-op", "--original-config-path", type=Path, default=None,
                            help="Path to the YAML/JSON file that contains the config for the original segmenter "
                                 "training Has to be provided if model was not trained and the original logging "
                                 "structure is not present, i.e. the config does not lie in a sibling directory of the "
                                 "checkpoint.")
    file_group.add_argument("-gt", "--ground-truth-dir", type=Path,
                            help="The Path to the directory in which the ground-truth images for segmentation are "
                                 "stored. This argument is required when trying to calculate the dice score using "
                                 "--calculate-dice-score.")
    file_group.add_argument("-o", "--output-dir", default="images", type=Path,
                            help="Path to the directory in which the results should be saved")
    file_group.add_argument("--handle-existing", default="abort", type=str, choices=["abort", "append", "overwrite"],
                            help="Determines what to do if there is already a results.json in the output directory.")

    # Preprocessing
    preprocessing_group = parser.add_argument_group("Input image preprocessing")
    preprocessing_group.add_argument("--resize", nargs=2, type=int,
                                     help="Resize input images to the given resolution [height width]. If one of the "
                                          "arguments is -1the size of this dimension will be determined "
                                          "automatically, keeping the aspect ratio.")
    preprocessing_group.add_argument("-bw", "--convert-to-black-white", action="store_true", default=False,
                                     help="If set the image will be converted to black/white before processing it.")

    # Setting hyperparameters
    hyperparam_group = parser.add_argument_group("Hyperparameter determination")
    overlap_group = hyperparam_group.add_mutually_exclusive_group()
    overlap_group.add_argument("--absolute-patch-overlap", nargs="+", type=int, default=[0],
                               help="Specify the overlap between patches in pixels: 0 < overlap < patch size. Cannot "
                                    "be used together with --path-overlap-factor.")
    overlap_group.add_argument("--patch-overlap-factor", nargs="+", type=float, default=[0.0],
                               help="Specify the overlap between patches as a percentage: 0.0 < overlap < 1.0. Cannot "
                                    "be used together with --absolute-patch-overlap.")
    hyperparam_group.add_argument("--min-confidence", nargs="+", type=float, default=[0.7],
                                  help="Specify the minimum confidence that is used in patch postprocessing. Pixels "
                                       "with a lower confidence than the specified values will be marked as "
                                       "background. 0.0 < min_confidence < 1.0. ")
    hyperparam_group.add_argument("--min-contour-area", nargs="+", type=int, default=[55],
                                  help="Specify the minimum contour area size that is used in patch postprocessing. "
                                       "Contours that are smaller than the specified values will be removed.")

    # Determine how the visual output should look like
    visual_output_group = parser.add_argument_group("Visual output determination")
    visual_output_group.add_argument("--extract-bboxes", action="store_true", default=False,
                                     help="extract bounding boxes of words by analyzing the found contours")
    visual_output_group.add_argument("--draw-patches", action="store_true", default=False,
                                     help="Show the borders of the patches, into which the image was cut")
    visual_output_group.add_argument("--draw-bboxes-on-segmentation", action="store_true", default=False,
                                     help="Draw the determined bounding boxes not only on original image but on the "
                                          "segmented image as well")
    visual_output_group.add_argument("-b", "--save-bboxes", action="store_true", default=False,
                                     help="Crop bounding boxes and save them as separate images")
    visual_output_group.add_argument("-c", "--save-contours", action="store_true", default=False,
                                     help="Crop contours and save them as separate images")
    visual_output_group.add_argument("--show-confidence", action="store_true", default=False,
                                     help="Reflects confidence of each prediction in the pixel color of the "
                                          "segmentation image. The lighter the color is the lower the confidence.")
    visual_output_group.add_argument("--overlay-segmentation", action="store_true", default=False,
                                     help="Overlays segmentation image over original image.")
    args = parser.parse_args()

    assert args.calculate_dice_score or args.visualize_segmentation, "Setting neither --calculate-dice-score nor " \
                                                                     "--visualize-segmentation will result in no " \
                                                                     "output."
    if args.calculate_dice_score:
        assert args.ground_truth_dir is not None, "If --calculate-dice-score is set --ground-truth-dir has to be set " \
                                                  "as well."
    return args


def create_hyperparam_configs(args):
    overlap = list(itertools.product(args.absolute_patch_overlap, args.patch_overlap_factor))
    hyperparam_combinations = list(itertools.product(args.min_confidence, args.min_contour_area, overlap))

    hyperparam_names = [("min_confidence", "min_contour_area", "patch_overlap")] * len(hyperparam_combinations)
    hyperparam_configs = tuple(map(lambda x, y: {k: v for k, v in zip(x, y)},
                                   hyperparam_names, hyperparam_combinations))
    return hyperparam_configs


def prepare_results(handle_existing: str, output_json_path: Path, model_config: dict, segmenter_config: dict,
                    class_to_color_map: dict) -> dict:
    if output_json_path.exists() and handle_existing != "overwrite":
        assert not handle_existing == "abort", f"{output_json_path} already exists and --handle-existing is set to " \
                                               "'abort'"
        if handle_existing == "append":
            with open(output_json_path, "r") as old_json:
                results = json.load(old_json)
            assert results["general_config"]["experiment_config"] == model_config, \
                "The previously saved experiment config does not match the current one. Use a new output dir instead " \
                "of setting --handle-existing to append."
            assert results["general_config"]["model_config"] == segmenter_config, \
                "The previously saved model config does not match the current one. Use a new output dir instead of " \
                "setting --handle-existing to append."
            assert results["general_config"]["class_to_color_map"] == class_to_color_map, \
                "The previously saved class to color map does not match the current one. Use a new output dir " \
                "instead of setting --handle-existing to append."
    else:
        results = {
            "general_config": {
                "experiment_config": model_config,
                "model_config": segmenter_config,
                "class_to_color_map": class_to_color_map
            },
            "runs": []
        }
    return results


def preprocess_images(image: ImageClass, args: argparse.Namespace) -> ImageClass:
    if args.resize:
        image = resize_image(image, args.resize)
    if args.convert_to_black_white:
        image = image.convert("L")
    return image


def get_string_representation_of_config(hyperparam_config):
    # Creating a string out of the hyperparameter configs that could be used as part of a filename. It concatenates
    # all key-value pairs in the hyperparam_config dictionary with underscores. Additionally, it removes parentheses
    # and replaces commas and whitespaces with underscores.
    # Example: {"min_confidence": 0, "min_contour_area": 10} becomes min_confidence_0_min_contour_area_10
    return "_".join([re.sub("[,\s.]", "_", re.sub("[()]", "", f"{k}_{v}")) for k, v in hyperparam_config.items()])


def main(args: argparse.Namespace) -> NoReturn:
    root_dir = Path(__file__).resolve().parent.parent
    with args.config_file.open() as f:
        model_config = json.load(f)
    segmenter = VotingAssemblySegmenter(
        model_config["checkpoint"],
        device="cuda",
        class_to_color_map=root_dir / model_config["class_to_color_map"],
        original_config_path=args.original_config_path,
        max_image_size=int(model_config.get("max_image_size", 0)),
        print_progress=False,
        show_confidence_in_segmentation=args.show_confidence
    )

    if "num_classes" in segmenter.config:
        num_classes = segmenter.config["num_classes"]
    else:
        # Fallback since earlier models didn't have the num_classes field
        num_classes = len(segmenter.class_to_color_map)
    class_to_color_map = segmenter.class_to_color_map
    class_names = list(class_to_color_map.keys())
    assert len(class_to_color_map) == num_classes, "Number of classes in color map and segmenter differs."

    hyperparam_configs = create_hyperparam_configs(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = args.output_dir / "results.json"

    scores_to_calculate = {"dice": args.calculate_dice_score, "iou": args.calculate_iou,
                           "precision": args.calculate_precision, "recall": args.calculate_recall}

    evaluate = any(scores_to_calculate.values())

    if evaluate:
        results = prepare_results(args.handle_existing, output_json_path, model_config, segmenter.config,
                                  class_to_color_map)
    else:
        print("No metrics specified, no evaluation will be run")

    image_paths = [f for f in args.image_dir.glob("**/*") if is_image(f)]
    assert len(image_paths) > 0, "There are no images in the given directory."
    for hyperparam_config in tqdm(hyperparam_configs, desc="Processing hyperparameter configs", leave=True):
        segmenter.set_hyperparams(hyperparam_config)

        if evaluate:
            results["runs"].append(defaultdict(dict))

        global_confusion_matrix = torch.zeros((num_classes, num_classes))

        for i, image_path in enumerate(tqdm(image_paths, desc="Processing images...", leave=False)):
            try:
                original_image = Image.open(image_path)
            except UnidentifiedImageError:
                print(f"File {image_path} is not an image.")
                continue
            image = preprocess_images(original_image, args)

            # The assembled prediction will be based on the post-processed softmax output of the network and the
            # assembly method (assemble_predictions) of the chosen segmenter type. For example, the standard
            # AnalysisSegmenter will save the highest confidences for each pixel from the different patches. However,
            # the VotingAssemblySegmenter will save the normalized, summed confidences over all the patches for each
            # pixel.
            assembled_prediction = segmenter.segment_image(image)

            if evaluate:
                try:
                    sample_confusion_matrix = calculate_confusion_matrix(assembled_prediction, image_path, args,
                                                                         class_to_color_map, num_classes)

                    sample_confusion_matrix_list = list(sample_confusion_matrix.reshape(-1).numpy().astype(float))
                    results["runs"][-1][f"confusion_matrices"][image_path.stem] = sample_confusion_matrix_list

                    global_confusion_matrix += sample_confusion_matrix

                    for metric_name, do_calculation in scores_to_calculate.items():
                        if do_calculation:
                            score = calculate_metric(sample_confusion_matrix, class_names, metric_name)
                            results["runs"][-1][f"detailed_{metric_name}_scores"][image_path.stem] = score
                except Exception as e:
                    print(f"The confusion matrix calculation produced an error:\n'{e}'\n"
                          f"The calculation for {image_path} will be skipped.\n")

            if args.visualize_segmentation:
                image_prefix = f"{image_path.stem}_{get_string_representation_of_config(hyperparam_config)}"
                try:
                    visualize_segmentation(assembled_prediction, image, original_image, segmenter, args,
                                           class_to_color_map, image_prefix)
                except Exception as e:
                    print(f"The visualization produced an error:\n'{e}'\n"
                          f"The visualization for {image_path} will be skipped.\n")

        for metric_name, do_calculation in scores_to_calculate.items():
            if do_calculation:
                average_score = calculate_metric(global_confusion_matrix, class_names, metric_name)
                results["runs"][-1][f"average_{metric_name}_scores"] = average_score

        if evaluate:
            results["runs"][-1]["hyperparams"] = hyperparam_config
            with open(output_json_path, "w") as out_json:
                json.dump(results, out_json, indent=4)


if __name__ == "__main__":
    print("Starting execution")
    main(parse_and_check_arguments())
