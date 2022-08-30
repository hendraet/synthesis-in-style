import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy
from PIL import Image
from csaps import csaps
from scipy import ndimage
from skimage.draw import line
from tqdm import tqdm, trange

from segmentation.evaluation.segmentation_visualization import draw_bounding_boxes
from utils.segmentation_utils import BBox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=Path, help='Path to image')
    parser.add_argument('meta_info_path', type=Path, help='Path to meta info JSON')
    return parser.parse_args()


def load_image_and_bboxes(args: argparse.Namespace) -> (Image.Image, Tuple[BBox, ...]):
    meta_information_path = args.meta_info_path
    with open(meta_information_path, 'r') as f:
        meta_information = json.load(f)
    segmentation_image = Image.open(args.image_path)
    assert tuple(
        meta_information['image_size']) == segmentation_image.size, 'Image size does not match meta information'

    bbox_dict = meta_information['bbox_dict']
    bboxes = tuple([BBox(*bbox) for bboxes in list(bbox_dict.values()) for bbox in bboxes])
    return segmentation_image, bboxes


def preprocess(segmentation_image: Image.Image, bboxes: Tuple[BBox, ...], padding: int = 10) -> (Image.Image, List[BBox]):
    top_left = [max(min(values) - padding, 0) for values in list(zip(*bboxes))[:2]]
    bottom_right = [max(values) + 10 for values in list(zip(*bboxes))[2:]]
    min_image = segmentation_image.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
    shifted_bboxes = numpy.asarray(bboxes) - numpy.asarray([*top_left, *top_left])
    return min_image, tuple((BBox(*bbox) for bbox in shifted_bboxes))


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


def get_dist_between_bbox_and_seam(bbox: BBox, seam: numpy.ndarray) -> int:
    bbox_x_mid = (bbox.left + bbox.right) // 2
    bbox_y_mid = (bbox.top + bbox.bottom) // 2
    dist = seam[bbox_x_mid, 0] - bbox_y_mid
    return dist


def map_bboxes_to_lines(bboxes: List[BBox], medial_seams: numpy.ndarray) -> Tuple[Dict[int, BBox], List[BBox]]:
    seams_y_min = medial_seams[:, :, 0].min(axis=1)
    seams_y_max = medial_seams[:, :, 0].max(axis=1)
    seams_y = numpy.stack((seams_y_min, seams_y_max), axis=1)

    # TODO: see if bboxes can be sorted to multiple seams or if we can detect if they should be split into two
    # Extract bounding boxes that intersect with at least one medial seam
    line_bbox_map = defaultdict(list)
    non_matched_bboxes = []
    for bbox in bboxes:
        final_line_id = None
        seam_candidates = numpy.where(numpy.logical_and(seams_y[:, 1] >= bbox.top, bbox.bottom >= seams_y[:, 0]))[0]
        for seam_id in seam_candidates:
            seam = medial_seams[seam_id]
            seam_part_y = seam[bbox.left:bbox.right, 0]
            if numpy.logical_and(seam_part_y >= bbox.top, seam_part_y <= bbox.bottom).any():
                # image.crop(bbox).show()
                # Calculate distance between bbox and seam so that the bbox is assigned to the closest seam
                dist = abs(get_dist_between_bbox_and_seam(bbox, seam))
                if final_line_id is None or dist < final_line_id[1]:
                    final_line_id = (seam_id, dist)

        if final_line_id is not None:
            line_bbox_map[final_line_id[0]].append(bbox)
        else:
            non_matched_bboxes.append(bbox)

    return line_bbox_map, non_matched_bboxes


def integrate_non_matched_bboxes(line_bbox_map, medial_seams, non_matched_bboxes):
    addtional_line_bbox_map = defaultdict(list)
    for non_matched_bbox in non_matched_bboxes:
        # draw_bounding_boxes(image, (non_matched_bbox,), outline_color=(0, 0, 0))  # TODO: remove

        dists_to_seams = [(i, abs(get_dist_between_bbox_and_seam(non_matched_bbox, seam))) for i, seam in
                          enumerate(medial_seams)]
        closest_seam_id = min(dists_to_seams, key=lambda x: x[1])[0]

        # TODO: remove
        # bbox_y_mid = (non_matched_bbox[1] + non_matched_bbox[3]) // 2  # TODO: maybe move
        # bbox_x_mid = (non_matched_bbox[0] + non_matched_bbox[2]) // 2
        # bbox_mid = (bbox_x_mid, bbox_y_mid)
        # seam_mid = tuple(reversed(medial_seams[closest_seam_id, bbox_x_mid]))
        # image_draw.line([bbox_mid, seam_mid], fill=(0, 0, 255), width=3)

        seam_bboxes = line_bbox_map[closest_seam_id]
        # get closest bbox in line
        # TODO: take more than 1 on each side?
        dists_left = [(seam_bbox, non_matched_bbox.left - seam_bbox[2]) for seam_bbox in seam_bboxes if
                      seam_bbox.right < non_matched_bbox[0]]
        bbox_left = min(dists_left, key=lambda x: x[1], default=(None,))[0]
        dists_right = [(seam_bbox, seam_bbox[0] - non_matched_bbox.right) for seam_bbox in seam_bboxes if
                       seam_bbox.left > non_matched_bbox[2]]
        bbox_right = min(dists_right, key=lambda x: x[1], default=(None,))[0]

        # TODO: remove
        # if bbox_left is not None:
        #     draw_bounding_boxes(image, (bbox_left,), outline_color=(0, 255, 0))
        # if bbox_right is not None:
        #     draw_bounding_boxes(image, (bbox_right,), outline_color=(255, 0, 0))

        if bbox_left is None or bbox_right is None:  # TODO: handle case where only one bbox is present
            continue
        neighboring_y_range = (min(bbox_left[1], bbox_right[1]), max(bbox_left[3], bbox_right[3]))

        bbox_y_mid = (non_matched_bbox[1] + non_matched_bbox[3]) // 2
        if neighboring_y_range[0] <= bbox_y_mid <= neighboring_y_range[1]:
            addtional_line_bbox_map[closest_seam_id].append(non_matched_bbox)
    return addtional_line_bbox_map


def do_ranges_overlap(range1, range2) -> bool:
    # TODO: remove if not used again
    return range2[1] >= range1[0] and range1[1] >= range2[0]


def merge_line_bbox_maps(line_bbox_map, addtional_line_bbox_map):
    new_line_bbox_map = defaultdict(list)
    for line_id, bboxes in line_bbox_map.items():
        new_line_bbox_map[line_id].extend(bboxes)
        if line_id in addtional_line_bbox_map:
            new_line_bbox_map[line_id].extend(addtional_line_bbox_map[line_id])
    return new_line_bbox_map


def split_lines(line_bbox_map: Dict[int, List[BBox]]) -> Dict[int, List[List[BBox]]]:
    partial_line_bbox_map = defaultdict(list)
    for line_id, line_bboxes in line_bbox_map.items():
        # calculate average bbox width in line
        sorted_line_boxes = sorted(line_bboxes, key=lambda bbox: bbox.left)
        avg_line_bbox_width = numpy.mean([bbox.width for bbox in sorted_line_boxes])
        current_line_part = []
        for i, line_bbox in enumerate(sorted_line_boxes):
            if i + 1 < len(sorted_line_boxes) and sorted_line_boxes[i + 1].left - line_bbox.right < 2 * avg_line_bbox_width:
                current_line_part.append(line_bbox)
            else:
                current_line_part.append(line_bbox)
                partial_line_bbox_map[line_id].append(current_line_part)
                current_line_part = []
    return partial_line_bbox_map


def main(args: argparse.Namespace):
    orig_image, orig_bboxes = load_image_and_bboxes(args)

    ### Preprocess image
    image, bboxes = preprocess(orig_image, orig_bboxes)
    r = 6  # TODO: dynamically calculate this, maybe a (roughly) fixed slice width?
    # TODO: check if this works with double page layouts. If not, try to find a fix
    medial_seams = calculate_medial_seams(image, r=r)

    ### Visualization
    ####

    line_bbox_map, non_matched_bboxes = map_bboxes_to_lines(bboxes, medial_seams)
    # TODO: try to bboxes that just slightly overlap line (e.g. "l" or "g")
    addtional_line_bbox_map = integrate_non_matched_bboxes(line_bbox_map, medial_seams, non_matched_bboxes)
    new_line_bbox_map = merge_line_bbox_maps(line_bbox_map, addtional_line_bbox_map)
    partial_line_bbox_map = split_lines(new_line_bbox_map)

    # TODO: when saving, maybe split into clean lines (no preprocessing such as splitting or inserting) and postprocess lines

    ### Visualize lines
    # TODO: remove
    from PIL import ImageDraw
    image_draw = ImageDraw.Draw(image)
    for medial_seam in medial_seams:
        points = [(x, y) for y, x in medial_seam]
        image_draw.line(points, fill=(0, 0, 255), width=5)
    for line_id, line_bboxes in addtional_line_bbox_map.items():
        draw_bounding_boxes(image, [BBox(b[0] - 3, b[1] - 3, b[2] + 3, b[3] + 3) for b in addtional_line_bbox_map[line_id]], outline_color=(153, 0, 204))
    for i, line_id in enumerate(partial_line_bbox_map.keys()):
        for part_id, line_part in enumerate(partial_line_bbox_map[line_id]):
            color = (255 if i % 2 == 0 else 0, 255 if part_id % 2 == 0 else 0, 0)
            draw_bounding_boxes(image, line_part, outline_color=color)
    image.show()
    ####
    print()


if __name__ == '__main__':
    main(parse_args())
