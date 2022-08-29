import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy
from PIL import Image
from csaps import csaps
from scipy import ndimage
from skimage.draw import line
from tqdm import tqdm, trange

from segmentation.evaluation.segmentation_visualization import draw_bounding_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=Path, help='Path to image')
    parser.add_argument('meta_info_path', type=Path, help='Path to meta info JSON')
    return parser.parse_args()


def load_image_and_bboxes(args: argparse.Namespace) -> (Image.Image, tuple):
    meta_information_path = args.meta_info_path
    with open(meta_information_path, 'r') as f:
        meta_information = json.load(f)
    segmentation_image = Image.open(args.image_path)
    assert tuple(
        meta_information['image_size']) == segmentation_image.size, 'Image size does not match meta information'

    bbox_dict = meta_information['bbox_dict']
    bboxes = tuple([bbox for bboxes in list(bbox_dict.values()) for bbox in bboxes])
    return segmentation_image, bboxes


def preprocess(segmentation_image: Image.Image, bboxes: tuple, padding: int = 10) -> (Image.Image, list):
    top_left = [max(min(values) - padding, 0) for values in list(zip(*bboxes))[:2]]
    bottom_right = [max(values) + 10 for values in list(zip(*bboxes))[2:]]
    min_image = segmentation_image.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
    shifted_bboxes = (numpy.asarray(bboxes) - numpy.asarray([*top_left, *top_left])).tolist()
    return min_image, shifted_bboxes


def calculate_maxima_locations(image_slice: numpy.ndarray, b: float) -> numpy.ndarray:
    slice_height, slice_width = image_slice.shape
    projection_profile = numpy.zeros((slice_height,))
    for i in range(slice_height):
        projection_profile[i] = numpy.sum(image_slice[i])
    x = numpy.linspace(0, len(projection_profile) - 1, num=len(projection_profile))
    smoothed_spline = csaps(x, projection_profile, smooth=b).spline
    d1 = smoothed_spline.derivative(nu=1)
    d2 = smoothed_spline.derivative(nu=2)
    extrema = d1.roots()
    maxima_locations = extrema[d2(extrema) < 0.0]

    return maxima_locations


def convert_maxima_to_medial_seams(fully_connected_slice_maxima: List[List[Tuple]],
                                   slice_widths: List[int]) -> numpy.ndarray:
    medial_seams = []
    for maxima_group in fully_connected_slice_maxima:
        medial_seam = []
        for slice_idx, (maximum, slice_width) in enumerate(zip(maxima_group[:-1], slice_widths)):
            next_maximum = maxima_group[slice_idx + 1]

            half_slice_width = slice_width // 2
            x_start = sum(slice_widths[:slice_idx]) + half_slice_width + 1

            next_half_slice_width = slice_widths[slice_idx + 1] // 2
            missing_slice_width = slice_width - half_slice_width
            x_end = x_start + missing_slice_width + next_half_slice_width

            y_coords, x_coords = line(maximum[0], x_start, next_maximum[0], x_end)
            medial_seam += list(zip(y_coords[:-1], x_coords[:-1]))

        # since we always draw lines from the middle of the slice we need to add padding for the first and last slice
        first_slice_half = [(medial_seam[0][0], x) for x in range(medial_seam[0][1])]
        last_slice_half = [(medial_seam[-1][0], x) for x in range(medial_seam[-1][1] + 1, sum(slice_widths))]
        medial_seam = first_slice_half + medial_seam + last_slice_half
        medial_seams.append(medial_seam)

    return numpy.asarray(medial_seams)


def calculate_medial_seams(image: Image.Image, r: int = 20, b: float = 0.0003) -> numpy.ndarray:
    grayscale_image = image.convert("L")
    image_array = numpy.asarray(grayscale_image)
    sobel_image = ndimage.sobel(image_array)
    slices = numpy.array_split(sobel_image, r, axis=1)

    # Calculate maxima for each slice
    slice_maxima = []
    for image_slice in tqdm(slices, desc="Calculate seam maxima...", leave=False):
        maxima_locations = calculate_maxima_locations(image_slice, b)
        slice_maxima.append(maxima_locations)

    # Match maxima locations across slices to extract seams
    connected_slice_maxima = {}  # maps the end point of a line to the list of points that are part of this line
    for slice_idx in trange(r - 1, desc="Matching maxima...", leave=False):
        for left_maximum_idx, left_maximum in enumerate(slice_maxima[slice_idx]):
            right_maxima = slice_maxima[slice_idx + 1]
            if len(right_maxima) == 0:
                continue

            dists_left_to_right = numpy.absolute(left_maximum - right_maxima)
            min_dist_idx_right = numpy.argmin(dists_left_to_right)
            right_maximum = right_maxima[min_dist_idx_right]
            dists_right_to_left = numpy.absolute(right_maximum - slice_maxima[slice_idx])
            min_dist_idx_left = numpy.argmin(dists_right_to_left)

            if min_dist_idx_left == left_maximum_idx:
                start_point = (int(round(left_maximum)), slice_idx)
                end_point = (int(round(right_maximum)), slice_idx + 1)
                if start_point not in connected_slice_maxima.keys():
                    connected_slice_maxima[end_point] = [start_point, end_point]
                else:
                    connected_slice_maxima[end_point] = connected_slice_maxima[start_point] + [end_point]
                    connected_slice_maxima.pop(start_point)

    fully_connected_slice_maxima = [v for k, v in connected_slice_maxima.items() if len(v) == r]

    slice_widths = [image_slice.shape[1] for image_slice in slices]
    medial_seams = convert_maxima_to_medial_seams(fully_connected_slice_maxima, slice_widths)

    return medial_seams


def main(args: argparse.Namespace):
    orig_image, orig_bboxes = load_image_and_bboxes(args)

    # draw_bounding_boxes(orig_image, orig_bboxes)
    # orig_image.show()

    ### Preprocess image
    image, bboxes = preprocess(orig_image, orig_bboxes)
    r = 6  # TODO: dynamically calculate this, maybe a (roughly) fixed slice width?
    # TODO: check if this works with double page layouts. If not, try to find a fix
    medial_seams = calculate_medial_seams(image, r=r)

    ### Visualization
    # draw_bounding_boxes(image, bboxes)
    from PIL import ImageDraw
    image_draw = ImageDraw.Draw(image)
    slice_width = image.width // r
    for i in range(r - 1):
        border_x = (i + 1) * slice_width
        patch_border = [(border_x, 0), (border_x, image.height - 1)]
        image_draw.line(patch_border, fill=(255, 0, 255), width=3)
    for medial_seam in medial_seams:
        points = [(x, y) for y, x in medial_seam]
        image_draw.line(points, fill=(0, 0, 255), width=5)
    # image.resize((image.width // 2, image.height // 2)).show()
    ####

    # TODO: maybe move to function
    seams_y_min = medial_seams[:, :, 0].min(axis=1)
    seams_y_max = medial_seams[:, :, 0].max(axis=1)
    seams_y = numpy.stack((seams_y_min, seams_y_max), axis=1)

    # TODO: see if bboxes can be sorted to multiple seams or if we can detect if they should be split into two
    line_bbox_map = defaultdict(list)
    for bbox in bboxes:
        final_line_id = None
        bbox_x_mid = (bbox[0] + bbox[2]) // 2
        bbox_y_mid = (bbox[1] + bbox[3]) // 2
        seam_candidates = numpy.where(numpy.logical_and(seams_y[:, 1] >= bbox[1], bbox[3] >= seams_y[:, 0]))[0]
        for seam_id in seam_candidates:
            seam = medial_seams[seam_id]
            seam_part_y = seam[bbox[0]:bbox[2], 0]
            if numpy.logical_and(seam_part_y >= bbox[1], seam_part_y <= bbox[3]).any():
                # image.crop(bbox).show()
                # Calculate distance between bbox and seam so that the bbox is assigned to the closest seam
                dist = abs(seam[bbox_x_mid, 0] - bbox_y_mid)
                if final_line_id is None or dist < final_line_id[1]:
                    final_line_id = (seam_id, dist)
        if final_line_id is not None:
            line_bbox_map[final_line_id[0]].append(bbox)
    # TODO: split lines in multiple parts if whitespaces in between are too large

    ### Visualize lines
    for i, (line_id, line_bboxes) in enumerate(line_bbox_map.items()):
        color = (255 if i % 3 == 0 else 0, 255 if i % 2 == 0 else 0, 0)
        draw_bounding_boxes(image, line_bboxes, outline_color=color)
    image.show()
    ####

    print()

    ###


if __name__ == '__main__':
    main(parse_args())
