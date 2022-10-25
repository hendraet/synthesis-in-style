import argparse
from pathlib import Path
from typing import List, Tuple

import imgaug.augmenters as iaa
import numpy
from PIL import Image
from pytorch_training.images import is_image
from tqdm import tqdm

def show_combined_images(original_image: numpy.ndarray, segmentation_image: numpy.ndarray,
                         augmented_images: numpy.ndarray, batch_size: int):
    unaugmented_images = numpy.concatenate((original_image, segmentation_image), axis=2)

    # Makes columns for each image in the original batch. Each column consists of the unaugmented image on top
    # followed by all the augmented images
    columns = []
    # Arrays have to be split first because they are still in batch form
    for unaug, aug in zip(numpy.split(unaugmented_images, batch_size), numpy.split(augmented_images, batch_size)):
        column = numpy.squeeze(numpy.concatenate([unaug] + numpy.split(aug, len(aug)), axis=1))
        columns.append(column)

    combined_image = numpy.concatenate(columns, axis=1)
    Image.fromarray(combined_image).show()


def augment_image(original_image: Image.Image, segmentation_image: Image.Image, num_images: int,
                  show_images: bool = False) -> List[Tuple[Image.Image, Image.Image]]:
    assert original_image.mode == original_image.mode, "The given images have different color modes."
    assert original_image.size == segmentation_image.size, \
        "Dimensions of original_image and segmentation_image do not match"

    geometric_aug = iaa.Sequential([
        iaa.SomeOf((1, 2), [
            iaa.ElasticTransformation(alpha=(5.0, 25.0), sigma=(5.0, 9.0)),
            iaa.ShearX((20, 20)),
            iaa.CropAndPad((-80, 80)),
            iaa.Affine(translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}),
        ]),
        iaa.Sometimes(0.66, iaa.OneOf([
            iaa.Rot90([1, 3]),
            iaa.Rotate((-15, 15))
        ])),
    ]).to_deterministic()

    color_aug = iaa.Sequential([
        iaa.Sometimes(0.8, iaa.OneOf([
            iaa.GammaContrast((1.5, 2.5)),  # darker
            iaa.GammaContrast((0.1, 1.0)),  # lighter
        ])),
        iaa.Sometimes(0.10, iaa.Invert()),
    ]).to_deterministic()

    # Duplicates each image num_images times so that in the end each image can be augmented individually
    inflated_original_image = numpy.stack([numpy.array(original_image) for _ in range(num_images)])
    inflated_segmentation_image = numpy.stack([numpy.array(segmentation_image) for _ in range(num_images)])

    augmented_original_batch = geometric_aug(images=color_aug(images=inflated_original_image))
    augmented_segmentation_batch = geometric_aug(images=inflated_segmentation_image)

    if show_images:
        augmented_image_arrays = numpy.concatenate((augmented_original_batch, augmented_segmentation_batch), axis=2)
        augmented_images = [numpy.squeeze(arr) for arr in numpy.split(augmented_image_arrays, num_images)]
        unaugmented_images = numpy.concatenate((original_image, segmentation_image), axis=1)
        combined_image = numpy.concatenate([unaugmented_images] + augmented_images)
        Image.fromarray(combined_image).show()

    augmented_original_images = [Image.fromarray(numpy.squeeze(arr))
                                 for arr in numpy.split(augmented_original_batch, num_images)]
    augmented_segmentation_images = [Image.fromarray(numpy.squeeze(arr))
                                     for arr in numpy.split(augmented_segmentation_batch, num_images)]
    return list(zip(augmented_original_images, augmented_segmentation_images))


def save_image_batches(augmented_images: List[Tuple[Image.Image, Image.Image]], image_path: Path,
                       dataset_dir: Path, save_dir: Path):
    image_sub_dir = image_path.relative_to(dataset_dir).parent
    new_sub_dir = save_dir / image_sub_dir
    if not new_sub_dir.exists():
        new_sub_dir.mkdir(parents=True)
    for image_idx, (orig_image, segmentation_image) in enumerate(augmented_images):
        output_image = Image.new('RGB', (2 * orig_image.width, orig_image.height))
        output_image.paste(orig_image, (0, 0))
        output_image.paste(segmentation_image, (orig_image.width, 0))
        output_image.save(new_sub_dir / f"{image_path.stem}_aug_{image_idx + 1}{image_path.suffix}")


def main(args: argparse.Namespace):
    image_paths = [f for f in args.dataset_dir.glob('**/*') if is_image(f)]
    for image_path in tqdm(image_paths[:10], desc="Processing image batches..."):
        image = Image.open(image_path)
        original_image = image.crop((0, 0, image.width // 2, image.height))
        segmentation_image = image.crop((image.width // 2, 0, image.width, image.height))
        augmented_images = augment_image(original_image, segmentation_image, num_images=args.num_augmented_images,
                                         show_images=args.show)
        if args.save_dir is not None:
            save_image_batches(augmented_images, image_path, args.dataset_dir, args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script that produces an augmented dataset that can be used for "
                                                 "document segmentation. Augmentation operations may include "
                                                 "ElasticTransform, Shear, Crop, Shifting, Rotation, GammaContrast "
                                                 "and color inversion.")
    parser.add_argument("dataset_dir", type=Path,
                        help="Path of the top level directory of the dataset that should be augmented.")
    parser.add_argument("-n", "--num-augmented-images", type=int, default=3,
                        help="How many augmented images should be produced for every input image")
    parser.add_argument("--show", action="store_true", default=False, help="Shows generated images")
    parser.add_argument("--save-dir", type=Path, help="Saves generated images to the given directory.")
    main(parser.parse_args())
