import argparse
import functools
import json
import random
import traceback
from pathlib import Path
from typing import Dict, Tuple, List, Union, Optional

import numpy
import os

import torch
from PIL import Image
from pytorch_training.images import make_image
from tqdm import tqdm

import global_config
from latent_projecting import Latents
from networks import load_autoencoder_or_generator, StyleganAutoencoder, TwoStemStyleganAutoencoder
from segmentation.base_dataset_segmenter import BaseDatasetSegmenter
from segmentation.black_white_handwritten_printed_text_segmenter import BlackWhiteHandwrittenPrintedTextDatasetSegmenter
from segmentation.dataset_gan_segmenter import DatasetGANSegmenter
from segmentation.evaluation.coco_gt import iter_through_images_in, COCOGtCreator
from utils.config import load_config
from utils.dataset_creation import get_base_dirs, build_latent_and_noise_generator, generate_images


def get_dataset_gan_params(autoencoder: Union[StyleganAutoencoder, TwoStemStyleganAutoencoder], mean_latent,
                           creation_config: dict, image_size: int, latent_size: int) -> dict:
    latent_code = torch.randn(1, latent_size)
    noise = autoencoder.decoder.make_noise()
    latent = Latents(latent_code, noise)

    activations, _ = generate_images(latent, autoencoder, mean_latent=mean_latent)

    feature_sizes = [ac.shape[1] for ac in activations.values()]
    unscaled_sizes = [ac.shape[-1] for ac in activations.values()]

    mode = "bilinear"
    upsamplers = []

    for size in unscaled_sizes:
        up = torch.nn.Upsample(scale_factor=image_size / size, mode=mode)
        upsamplers.append(up.cuda())

    creation_config['feature_size'] = sum(feature_sizes)
    creation_config['upsamplers'] = upsamplers

    return creation_config


def get_dataset_segmenter(args: argparse.Namespace, creation_config: dict, image_size: int,
                          semantic_segmentation_base_dir: Path) -> BaseDatasetSegmenter:
    if creation_config['segmenter_type'] == 'black_white_handwritten_printed':
        assert 'only_keep_overlapping' in creation_config, 'The key "only_keep_overlapping" must be ' \
                                                           'specified in the config file.'
        segmenter_class = functools.partial(
            BlackWhiteHandwrittenPrintedTextDatasetSegmenter,
            keys_to_merge=creation_config['keys_to_merge'],
            only_keep_overlapping=creation_config['only_keep_overlapping'],
            keys_for_class_determination=creation_config['keys_for_class_determination'],
            keys_for_finegrained_segmentation=creation_config['keys_for_finegrained_segmentation'],
            num_clusters=args.num_clusters,
            min_class_contour_area=creation_config['min_class_contour_area']
        )
    elif creation_config['segmenter_type'] == 'dataset_gan':
        segmenter_class = functools.partial(
            DatasetGANSegmenter,
            classifier_path=args.classifier_path,
            feature_size=creation_config['feature_size'],
            upsamplers=creation_config['upsamplers']
        )
    else:
        raise NotImplementedError

    segmenter = segmenter_class(
        base_dir=semantic_segmentation_base_dir,
        image_size=image_size,
        class_to_color_map=creation_config['class_to_color_map'],
    )
    return segmenter


def save_image(image: numpy.ndarray, image_id: int, base_dir: Path, name_format: str = "{id}.png"):
    save_sub_folder_1 = str(image_id // 1000)
    save_sub_folder_2 = str(image_id // 100000)
    dest_file_name = base_dir / save_sub_folder_2 / save_sub_folder_1 / name_format.format(id=image_id)
    dest_file_name.parent.mkdir(exist_ok=True, parents=True)
    image = Image.fromarray(image)
    image.save(str(dest_file_name))


def save_generated_images(generated_images: numpy.ndarray, semantic_segmentation_images: numpy.ndarray, batch_id: int,
                          base_dir: Path, num_images: int):
    images = numpy.concatenate([generated_images, semantic_segmentation_images], axis=2)

    for idx, image in enumerate(images):
        image_id = batch_id + idx
        save_image(image, image_id, base_dir, name_format=f"{{id:0{max(4, len(str(num_images)))}d}}.png")


def save_debug_images(debug_images: Dict[str, List[numpy.ndarray]], iteration: int, base_dir: Path):
    for batch_id in range(len(list(debug_images.values())[0])):
        image = numpy.concatenate([images[batch_id] for images in debug_images.values()], axis=1)
        image_id = iteration + batch_id
        save_image(image, image_id, base_dir, name_format=f"{{id:04d}}_debug.png")


def build_dataset(args: argparse.Namespace, creation_config: Dict, original_config_path: Optional[Path] = None):
    config = load_config(args.checkpoint, original_config_path)
    config['batch_size'] = args.batch_size
    image_save_base_dir, semantic_segmentation_base_dir = get_base_dirs(args)
    autoencoder = load_autoencoder_or_generator(args, config)

    if args.truncate:
        mean_latent = autoencoder.decoder.mean_latent(4096)
    else:
        mean_latent = None

    if creation_config['segmenter_type'] == 'dataset_gan':
        creation_config = get_dataset_gan_params(autoencoder, mean_latent, creation_config,
                                                 config['image_size'], config['latent_size'])

    data_loader = build_latent_and_noise_generator(autoencoder, config, seed=creation_config['seed'])
    data_iter = iter(data_loader)

    segmenter = get_dataset_segmenter(args, creation_config, config['image_size'], semantic_segmentation_base_dir)

    with tqdm(total=args.num_images, desc="Creating images") as pbar:
        while pbar.n < args.num_images:
            batch = next(data_iter)
            activations, generated_images = generate_images(batch, autoencoder, mean_latent=mean_latent)
            semantic_label_images, image_ids_to_drop = segmenter.create_segmentation_image(activations)

            generated_images = make_image(generated_images)
            if not global_config.debug:
                generated_images = numpy.delete(generated_images, image_ids_to_drop, axis=0)
                semantic_label_images = numpy.delete(semantic_label_images, image_ids_to_drop, axis=0)

            num_generated_images = len(semantic_label_images)
            if num_generated_images > 0:
                save_generated_images(generated_images, semantic_label_images, pbar.n, image_save_base_dir,
                                      args.num_images)
            if global_config.debug and len(segmenter.debug_images) > 0:
                save_debug_images(segmenter.debug_images, pbar.n, image_save_base_dir)
                pbar.update(args.batch_size)
            else:
                pbar.update(num_generated_images)


def create_dataset_json_data(image_paths: List[Path], image_root: Path, gt_creator: COCOGtCreator) \
        -> Tuple[List[dict], bool]:
    dataset_data = []
    try:
        for image_path in tqdm(image_paths, desc='create dataset json data', leave=False, unit='img'):
            with Image.open(str(image_path)) as the_image:
                data = {
                    "file_name": str(image_path.relative_to(image_root)),
                }
                data.update(gt_creator.determine_classes_in_image(the_image))
            dataset_data.append(data)
    except:
        print(traceback.format_exc())
        return dataset_data, False

    return dataset_data, True


def main(args: argparse.Namespace):
    with open(args.config) as f:
        config = json.load(f)

    if not args.only_create_train_val_split:
        build_dataset(args, config, original_config_path=args.original_config_path)

    if global_config.debug:
        # no need for gt if only creating debug images
        return

    image_save_base_dir, _ = get_base_dirs(args)
    generated_images = list(iter_through_images_in(image_save_base_dir))
    random.seed(config['seed'])
    random.shuffle(generated_images)

    coco_creator = COCOGtCreator(config['class_to_color_map'], image_root=image_save_base_dir)

    # 10% validation data
    split_index = int(len(generated_images) * 0.9)
    training_images = generated_images[:split_index]
    validation_images = generated_images[split_index:]

    training_gt, success = create_dataset_json_data(training_images, image_save_base_dir, coco_creator)
    train_filename = image_save_base_dir / ('train.json' if success else 'train.json.part')
    with train_filename.open('w') as f:
        json.dump(training_gt, f)
    del training_gt

    validation_gt, success = create_dataset_json_data(validation_images, image_save_base_dir, coco_creator)
    val_filename = image_save_base_dir / ('val.json' if success else 'val.json.part')
    with val_filename.open('w') as f:
        json.dump(validation_gt, f)
    del validation_gt

    coco_gt = coco_creator.create_coco_gt_from_image_paths(validation_images)
    with (image_save_base_dir / 'coco_gt.json').open('w') as f:
        json.dump(coco_gt, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset using a trained StyleGAN model and the "
                                                 "labelled intermediate layers specified in a config file.")
    parser.add_argument("checkpoint", help="Path to trained autoencoder/generator for dataset creation")
    parser.add_argument("config", help="path to json file containing config for generation")
    parser.add_argument("-op", "--original-config-path", type=Path, default=None,
                        help="Path to the YAML/JSON file that contains the config for the original segmenter "
                             "training Has to be provided if model was not trained and the original logging "
                             "structure is not present, i.e. the config does not lie in a sibling directory of the "
                             "checkpoint.")
    parser.add_argument("-n", "--num-images", type=int, default=100, help="Number of images to generate")
    parser.add_argument("-s", "--save-to",
                        help="path where to save generated images (default is save in dir of run of used checkpoint)")
    parser.add_argument("-b", "--batch-size", default=10, type=int, help="batch size for generation of images on GPU")
    parser.add_argument("-d", "--device", default='cuda',
                        help="CUDA device to use, either any (cuda) or the id of the device")
    parser.add_argument("--only-create-train-val-split", action='store_true', default=False,
                        help="do not create an entire dataset, rather use the save_path and build a train validation "
                             "split with according COCO GT")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="render debug output during image generation")
    parser.add_argument("--truncate", action='store_true', default=False, help="Use truncation trick during generation")
    parser.add_argument("--num-clusters", type=int, default=-1,
                        help="The number of classes labelled with semantic labeler. Only used with cluster-based "
                             "segmenters.")
    parser.add_argument("--classifier-path", help="Path to the trained activation classifier. Only used with "
                                                  "DatasetGAN segmenters.")
    parser.add_argument("-ssd", "--semantic-segmentation-base-dir", type=Path,
                        help="If a different directory for creating the semantic segmentation was chosen use this flag "
                             "to provide it")

    parsed_args = parser.parse_args()
    global_config.debug = parsed_args.debug
    main(parsed_args)
