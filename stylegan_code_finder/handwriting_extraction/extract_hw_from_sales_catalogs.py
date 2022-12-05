import argparse
import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from stylegan_code_finder.handwriting_extraction.get_lines_from_segementation_bboxes import process_segmentation, \
    set_opt_args_for_hw_extraction
from stylegan_code_finder.segmentation.analysis_segmenter import VotingAssemblySegmenter
from stylegan_code_finder.segmentation.evaluation.segmentation_visualization import draw_segmentation
from stylegan_code_finder.utils.config import load_yaml_config

if os.environ.get('REMOTE_PYCHARM_DEBUG_SESSION', False):
    import pydevd_pycharm

    pydevd_pycharm.settrace('localhost', port=int(os.environ.get('REMOTE_PYCHARM_DEBUG_PORT')),
                            stdoutToServer=True, stderrToServer=True, suspend=False)
    print('Remote debugger is up and running')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('txt_path', type=Path, help='Txt file that lists all files and if they contain handwriting')
    parser.add_argument('model_config_path', type=Path, help='JSON containing the model configuration')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    set_opt_args_for_hw_extraction(parser)
    return parser.parse_args()


def get_segmenter(model_config: dict, root_dir: Path = Path('../scripts'), original_config_path: Optional[Path] = None,
                  batch_size: Optional[int] = None, show_confidence: bool = False) -> VotingAssemblySegmenter:
    segmenter = VotingAssemblySegmenter(
        model_config["checkpoint"],
        device="cuda",
        class_to_color_map=root_dir / model_config["class_to_color_map"],
        original_config_path=original_config_path,
        batch_size=batch_size,
        max_image_size=int(model_config.get("max_image_size", 0)),
        print_progress=True,
        show_confidence_in_segmentation=show_confidence
    )
    return segmenter


def save_segmentation_json(json_out_path, bbox_dict, hyperparams, checkpoint_path, model_name, image_size):
    meta_information = {
        'model_name': model_name,
        'model_checkpoint': checkpoint_path,
        'model_hyperparams': hyperparams,
        'image_size': image_size,
        'bbox_dict': bbox_dict
    }
    with json_out_path.open('w') as f:
        json.dump(meta_information, f)


def main(args: argparse.Namespace):
    output_root_dir = Path('/home/hendrik/wpi-gan-generator-project/datasets/debug/extract_hw_test_run')
    output_root_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    error_log_path = output_root_dir / f'error_log_{timestamp}.txt'
    logging.basicConfig(filename=error_log_path, level=logging.WARNING, format='%(asctime)s:%(levelname)s: %(message)s')

    model_config = load_yaml_config(args.model_config_path)
    # segmenter = get_segmenter(model_config, show_confidence=True)
    segmenter = get_segmenter(model_config, batch_size=args.batch_size,
                              show_confidence=getattr(model_config, "show_confidence", False))

    # High Precision settings
    hyperparameters = {
        'trans_u_net': {
            'min_confidence': 0.9,
            'min_contour_area':	15,
            'patch_overlap': [0, 0.5],
        },
    }

    hyperparams = hyperparameters[model_config['model_name']]
    segmenter.set_hyperparams(hyperparams)
    class_to_color_map = segmenter.class_to_color_map
    class_to_id_map = {class_name: i for i, class_name in enumerate(class_to_color_map.keys())}
    filter_classes = (class_to_id_map['printed_text'],)

    with args.txt_path.open() as f:
        lines = [line.strip() for line in f]

    for line in tqdm(lines[12:15], desc='Processing images'):
        image_path = args.txt_path.parent / line
        # image_path = Path('/dataset/sales_cat/00000761/010000006782654_002/010000006782654_002_0030.jpg')
        # image_path = Path('/home/hendrik/wpi-gan-generator-project/datasets/debug/debug_hw_lines.png')
        # image_path = Path('/home/hendrik/wpi-gan-generator-project/datasets/debug/010000006782654_002_0030_rotated.png')
        # image_path = Path('/home/hendrik/wpi-gan-generator-project/datasets/debug/debug_hw_difficult_lines_small.png')
        image_path = Path('/home/hendrik/pipeline-project/src/test/test_image_2_medium.png')

        if not image_path.exists():
            logging.warning(f'{image_path} does not exist')
            continue

        try:
            original_image = Image.open(image_path)
        except UnidentifiedImageError:
            logging.error(f"File {image_path} is not an image.")
            continue
        image = original_image.convert('L')

        assembled_prediction = segmenter.segment_image(image)
        segmented_image = segmenter.prediction_to_color_image(assembled_prediction)

        output_dir = output_root_dir / Path(line).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        image_prefix = f'{image_path.stem}_{model_config["model_name"]}'
        segmented_image.save(output_dir / f"{image_prefix}_segmented_no_bboxes.png")

        ### vis
        image_w_bboxes, segmented_image_w_bboxes, bbox_dict = draw_segmentation(
            original_image,
            assembled_prediction,
            segmented_image,
            bboxes_for_patches=None,
            filter_classes=filter_classes,
            return_bboxes=True
        )
        # TODO: create flags to decide if these should be saved
        image_w_bboxes.save(output_dir / f"{image_prefix}_bboxes.png")
        segmented_image_w_bboxes.save(output_dir / f"{image_prefix}_segmented.png")
        ###

        id_to_class_map = {v: k for k, v in class_to_id_map.items()}
        bbox_dict = {id_to_class_map[k]: v for k, v in bbox_dict.items()}

        json_out_path = output_dir / f"{image_prefix}_meta_raw.json"
        save_segmentation_json(json_out_path, bbox_dict, hyperparams, model_config['checkpoint'],
                               model_config['model_name'], image.size)

        try:
            process_segmentation(
                bbox_dict=bbox_dict,
                segmented_image=segmented_image,
                original_image=original_image,
                slice_width=args.slice_width,
                original_image_name=Path(Path(line).name),
                out_dir=output_dir,
                keep_ambiguous_lines=args.save_ambiguous_lines,
                min_num_bboxes=args.min_num_bboxes,
                min_aspect_ratio=args.min_aspect_ratio,
                min_line_area=args.min_line_area,
                debug=args.debug
            )
        except Exception as e:
            logging.error(f'{image_path}: {traceback.format_exc()}')
        break


if __name__ == '__main__':
    main(parse_args())
