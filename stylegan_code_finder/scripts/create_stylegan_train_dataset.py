import argparse
import json
import math
import random
from pathlib import Path
from typing import List

import cv2
import numpy
from PIL import Image
from PIL.Image import Image as ImageClass
from tqdm import tqdm
from tqdm.contrib import tenumerate

from pytorch_training.images import is_image


def crop_patches(image: ImageClass, image_size: int) -> List[ImageClass]:
    windows_in_width = math.ceil(image.width / image_size)
    total_width_overlap = windows_in_width * image_size - image.width
    windows_in_height = math.ceil(image.height / image_size)
    total_height_overlap = windows_in_height * image_size - image.height

    width_overlap_per_patch = total_width_overlap / windows_in_width
    height_overlap_per_patch = total_height_overlap / windows_in_height

    patches = []
    for y_idx in range(windows_in_height):
        start_y = y_idx * (image_size - height_overlap_per_patch)
        for x_idx in range(windows_in_width):
            start_x = x_idx * (image_size - width_overlap_per_patch)
            patches.append(image.crop((start_x, start_y, start_x + image_size, start_y + image_size)))

    return patches


def random_resize(image: ImageClass, min_size: int = 1000) -> ImageClass:
    downsample_factor = random.randint(1, 4)
    new_width = image.width / downsample_factor
    new_height = image.height / downsample_factor
    new_size = max(new_height, new_width)
    if new_size < min_size:
        new_size = min_size

    image.thumbnail((new_size, new_size))
    return image


def get_content_box(the_image, edge_detect=True):
    if edge_detect:
        image = numpy.array(the_image)
        image = cv2.blur(image, (3, 3))
        thresh = cv2.Canny(image, 20, 150)
        erode_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        thresh = cv2.dilate(thresh, erode_dilate_kernel)
        thresh = cv2.erode(thresh, erode_dilate_kernel, 2)
    else:
        gray = the_image.convert('L')
        gray = numpy.array(gray)
        ret, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [
            0,
            0,
            the_image.width,
            the_image.height
        ]
    hierarchy = hierarchy.reshape(-1, hierarchy.shape[-1])
    if len(hierarchy) == 1:
        return [
            0,
            0,
            the_image.width,
            the_image.height
        ]

    # sort contours by contour with largest area
    contours = [(contour, (rect := cv2.boundingRect(contour))[2] * rect[3]) for contour in contours]
    # uncomment this for non Python 3.8
    # contours = [(contour, cv2.boundingRect(contour)) for contour in contours]
    # contours = [(contour[0], contour[1][2] * contour[1][3]) for contour in contours]

    contours = list(sorted(contours, key=lambda x: x[1], reverse=True))
    differences = [abs(contour[1] - next_contour[1]) for contour, next_contour in zip(contours, contours[1:])]
    contours, areas = list(zip(*contours))

    largest_area = areas[0]
    image_area = thresh.size
    if image_area * 0.6 > largest_area:
        # if the area we found is not large enough, we assume that there is no scanning margin!
        return [
            0,
            0,
            the_image.width,
            the_image.height
        ]

    # throw away all contours that follow the largest difference -> keep the largest that are similar in size and remove
    # all small contours
    max_difference_index = differences.index(max(differences))

    contours = contours[:max_difference_index + 1]
    content_bounding_box = cv2.boundingRect(numpy.concatenate(contours, axis=0))
    return [
        content_bounding_box[0],
        content_bounding_box[1],
        content_bounding_box[0] + content_bounding_box[2],
        content_bounding_box[1] + content_bounding_box[3],
    ]


def scale_bounding_box(box, box_image_extent, new_image_extent):
    box_image_width, box_image_height = box_image_extent
    new_image_width, new_image_height = new_image_extent

    width_factor = new_image_width / box_image_width
    height_factor = new_image_height / box_image_height

    new_box = [
        box[0] * width_factor,
        box[1] * height_factor,
        box[2] * width_factor,
        box[3] * height_factor,
    ]

    return list(map(int, new_box))


def remove_scanning_margin(the_image: ImageClass) -> ImageClass:
    analysis_image = the_image.copy()
    analysis_image.thumbnail((1000, 1000))
    content_bounding_box = get_content_box(analysis_image)

    crop_box = scale_bounding_box(content_bounding_box, analysis_image.size, the_image.size)
    crop = the_image.crop(crop_box)
    return crop


def main(args: argparse.Namespace):
    root_dir = Path(args.root_dir)
    destination = Path(args.destination)
    destination.mkdir(exist_ok=True, parents=True)

    if not args.only_jsons:
        if args.filter is not None:
            glob_filter = f"**/*{args.filter}*"
        else:
            glob_filter = "**/*"
        files_in_root = [file_name for file_name in root_dir.glob(glob_filter) if is_image(file_name)]
        num_files_to_use = min(len(files_in_root), args.max_num_samples)
        random.shuffle(files_in_root)

        patch_paths = []
        for idx, file_path in tenumerate(files_in_root, total=num_files_to_use):
            file_dir = file_path.parent
            relative_path_in_root = file_dir.relative_to(root_dir)
            dest_dir = destination / relative_path_in_root
            dest_dir.mkdir(exist_ok=True, parents=True)
            try:
                with Image.open(str(file_path)) as the_image:
                    if args.margin_remove:
                        the_image = remove_scanning_margin(the_image)

                    if any(side > args.max_size for side in the_image.size):
                        the_image.thumbnail((args.max_size, args.max_size))

                    the_image = random_resize(the_image)
                    patches = crop_patches(the_image, args.image_size)
                    for patch_idx, patch in tenumerate(patches, leave=False):
                        patch_file_name = dest_dir / f"{file_path.stem}_{patch_idx}.png"
                        patch.save(str(patch_file_name))
                        patch_paths.append(str(patch_file_name.relative_to(destination)))
                idx += 1
            except Exception as e:
                print(e)
            if idx >= num_files_to_use:
                break
    else:
        patch_paths = [file_name for file_name in destination.glob('**/*') if is_image(file_name)]
        patch_paths = patch_paths[:args.max_num_samples]
        patch_paths = [str(file_name.relative_to(destination)) for file_name in patch_paths]

    # 10% percent validation split
    random.shuffle(patch_paths)
    split_index = int(len(patch_paths) * 0.9)
    train_patches = patch_paths[:split_index]
    validation_patches = patch_paths[split_index:]

    with (destination / 'train.json').open('w') as f:
        json.dump(train_patches, f)

    with (destination / 'val.json').open('w') as f:
        json.dump(validation_patches, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that crops parts from images to be used to train StyleGAN")
    parser.add_argument("root_dir", help="root dir of original images")
    parser.add_argument("destination", help="path to destination where resulting images shall be saved")
    parser.add_argument("max_num_samples", type=int, help="maximum number of samples to crop patches from")
    parser.add_argument("--image-size", type=int, default=256, help="size of crops to extract from original images")
    parser.add_argument("--only-jsons", action='store_true', default=False, help="do not copy images, but create json from files in dest dir")
    parser.add_argument("--max-size", type=int, default=3000, help="max size of images before patches are cropped")
    parser.add_argument("--margin-remove", action='store_true', default=False, help="Run margin removal to remove scanning margins before generation of crops")
    parser.add_argument("--filter", help="filter expression to use when loading all possible images")

    main(parser.parse_args())
