import json
import logging

import os

import argparse
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from segmentation.analysis_segmenter import VotingAssemblySegmenter
from segmentation.evaluation.analyze_image_segments import preprocess_images
from segmentation.evaluation.segmentation_visualization import extract_and_save_bounding_boxes, visualize_segmentation, \
    draw_segmentation
from utils.config import load_yaml_config

if os.environ.get('REMOTE_PYCHARM_DEBUG_SESSION', False):
    import pydevd_pycharm

    pydevd_pycharm.settrace('localhost', port=int(os.environ.get('REMOTE_PYCHARM_DEBUG_PORT')),
                            stdoutToServer=True, stderrToServer=True, suspend=False)
    print('Remote debugger is up and running')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('txt_path', type=Path, help='Txt file that lists all files and if they contain handwriting')
    parser.add_argument('model_config_path', type=Path, help='JSON containing the model configuration')
    return parser.parse_args()


def get_segmenter(model_config: dict, root_dir: Path = Path('.'), original_config_path: Optional[Path] = None,
                  show_confidence: bool = False) -> VotingAssemblySegmenter:
    segmenter = VotingAssemblySegmenter(
        model_config["checkpoint"],
        device="cuda",
        class_to_color_map=root_dir / model_config["class_to_color_map"],
        original_config_path=original_config_path,
        max_image_size=int(model_config.get("max_image_size", 0)),
        print_progress=True,
        show_confidence_in_segmentation=show_confidence  # TODO: plot conf
    )
    return segmenter


def main(args: argparse.Namespace):
    # TODO: also extract meta information such as location and maybe also confidences?
    # TODO: find out if rotated text can/should be extracted (90° vs slight rotations)
    model_config = load_yaml_config(args.model_config_path)
    # segmenter = get_segmenter(model_config, show_confidence=True)
    segmenter = get_segmenter(model_config, show_confidence=getattr(model_config, "show_confidence", False))

    # High recall settings
    #
    hyperparameters = {  # TODO: move to config?
        'doc_ufcn': {
            'min_confidence': 0.3,
            'min_contour_area':	15,
            'patch_overlap': [0, 0.5],  # TODO: maybe change to 0.0 for faster debugging
        },
        'trans_u_net': {
            'min_confidence': 0.9,
            'min_contour_area':	15,
            'patch_overlap': [0, 0.5],
        },
        'ema_net':  {
            'min_confidence': 0.3,
            'min_contour_area':	15,
            # 'patch_overlap': [0, 0.5]  # TODO:
            'patch_overlap': [0, 0.0]
        },
        'doc_ufcn_best': {
            "patch_overlap": [0, 0.5],
            "min_confidence": 0.7,
            "min_contour_area": 55
        }
    }

    hyperparams = hyperparameters[model_config['model_name']]
    segmenter.set_hyperparams(hyperparams)
    class_to_color_map = segmenter.class_to_color_map
    class_to_id_map = {class_name: i for i, class_name in enumerate(class_to_color_map.keys())}
    filter_classes = (class_to_id_map['printed_text'],)

    with args.txt_path.open() as f:
        lines = [line.strip() for line in f]

    for line in tqdm(lines[2:4], desc='Processing images'):  # TODO no limit
        image_path = args.txt_path.parent / line
        # image_path = Path('/dataset/sales_cat/00000761/010000006782654_002/010000006782654_002_0030.jpg')  # TODO: remove
        image_path = Path('/home/hendrik/wpi-gan-generator-project/datasets/debug/debug_hw_lines.png')  # TODO: remove
        # image_path = Path('/home/hendrik/wpi-gan-generator-project/datasets/debug/debug_hw_difficult_lines_small.png')  # TODO: remove
        if not image_path.exists():
            logging.warning(f'{image_path} does not exist')
            continue

        try:
            original_image = Image.open(image_path)
        except UnidentifiedImageError:
            print(f"File {image_path} is not an image.")
            continue
        # image = preprocess_images(original_image, args)  # TODO: maybe also resize
        image = original_image.convert('L')

        assembled_prediction = segmenter.segment_image(image)

        image_prefix = f'{image_path.stem}_{model_config["model_name"]}'
        image_save_dir = Path('/home/hendrik/wpi-gan-generator-project/datasets/debug/extract_hw')  # TODO: magic string
        image_save_dir.mkdir(exist_ok=True, parents=True)

        segmented_image = segmenter.prediction_to_color_image(assembled_prediction)
        segmented_image.save(image_save_dir / f"{image_prefix}_segmented_no_bboxes.png")

        ### vis
        image_w_bboxes, segmented_image_w_bboxes, bbox_dict = draw_segmentation(
            original_image,
            assembled_prediction,
            segmented_image,
            bboxes_for_patches=None,
            filter_classes=filter_classes,
            return_bboxes=True
        )
        image_w_bboxes.save(image_save_dir / f"{image_prefix}_bboxes.png")
        segmented_image_w_bboxes.save(image_save_dir / f"{image_prefix}_segmented.png")
        ###

        id_to_class_map = {v: k for k, v in class_to_id_map.items()}
        bbox_dict = {id_to_class_map[k]: v for k, v in bbox_dict.items()}

        meta_information = {
            'model_name': model_config['model_name'],
            'model_checkpoint': model_config['checkpoint'],
            'hyperparams': hyperparams,
            'image_size': image.size,
            'bbox_dict': bbox_dict
        }
        with (image_save_dir / f"{image_prefix}_meta.json").open('w') as f:
            json.dump(meta_information, f)

        break # TODO: remove


if __name__ == '__main__':
    main(parse_args())
