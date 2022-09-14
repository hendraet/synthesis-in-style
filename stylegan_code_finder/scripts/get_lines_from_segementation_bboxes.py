import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy
from PIL import Image
from csaps import csaps
from scipy import ndimage
from skimage.draw import line
from tqdm import tqdm, trange

from segmentation.evaluation.segmentation_visualization import draw_bounding_boxes
from utils.segmentation_utils import BBox


class LineBBox:
    # TODO: maybe move to extra file
    def __init__(self, left: int, top: int, right: int, bottom: int, line_candidates: Optional[List[int]] = None):
        self.bbox = BBox(left, top, right, bottom)
        self.line_candidates = line_candidates if line_candidates is not None else []

    def is_ambiguous(self) -> bool:
        return len(self.line_candidates) > 1

    @property
    def left(self) -> int:
        return self.bbox.left

    @property
    def right(self) -> int:
        return self.bbox.right

    @property
    def top(self) -> int:
        return self.bbox.top

    @property
    def bottom(self) -> int:
        return self.bbox.bottom

    @property
    def width(self) -> int:
        return self.bbox.width

    @property
    def height(self) -> int:
        return self.bbox.height

    def __iter__(self):
        return iter(self.bbox)

    def __str__(self):
        return f"LineBBox(({self.left}, {self.top}, {self.right}, {self.bottom}), {self.line_candidates})"

    def __repr__(self):
        return f"LineBBox({self.left}, {self.top}, {self.right}, {self.bottom}, {self.line_candidates})"




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=Path, help='Path to image')
    parser.add_argument('meta_info_path', type=Path, help='Path to meta info JSON')
    return parser.parse_args()


def load_image_and_bboxes(args: argparse.Namespace) -> (Image.Image, Tuple[LineBBox, ...]):
    meta_information_path = args.meta_info_path
    with open(meta_information_path, 'r') as f:
        meta_information = json.load(f)
    segmentation_image = Image.open(args.image_path)
    assert tuple(
        meta_information['image_size']) == segmentation_image.size, 'Image size does not match meta information'

    bbox_dict = meta_information['bbox_dict']
    bboxes = tuple([LineBBox(*bbox) for bboxes in list(bbox_dict.values()) for bbox in bboxes])
    return segmentation_image, bboxes


def preprocess(segmentation_image: Image.Image, bboxes: Tuple[LineBBox, ...], padding: int = 10) -> (Image.Image, List[LineBBox]):
    top_left = [max(min(values) - padding, 0) for values in list(zip(*bboxes))[:2]]
    bottom_right = [max(values) + 10 for values in list(zip(*bboxes))[2:]]
    min_image = segmentation_image.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
    shifted_bboxes = numpy.asarray([b.bbox for b in bboxes]) - numpy.asarray([*top_left, *top_left])
    return min_image, tuple((LineBBox(*bbox) for bbox in shifted_bboxes))


def calculate_maxima_locations(image_slice: numpy.ndarray, b: float) -> numpy.ndarray:
    slice_height, slice_width = image_slice.shape
    projection_profile = numpy.zeros((slice_height,))
    for i in range(slice_height):
        projection_profile[i] = numpy.sum(image_slice[i])
    x = numpy.linspace(0, len(projection_profile) - 1, num=len(projection_profile))
    smoothed_spline = csaps(x, projection_profile, smooth=b).spline
    d1 = smoothed_spline.derivative(nu=1)
    d2 = smoothed_spline.derivative(nu=2)
    extrema = d1.roots().astype(int)
    maxima = extrema[d2(extrema) < 0.0]
    maxima = maxima[numpy.logical_and(maxima > 0, maxima < image_slice.shape[0])]  # remove out of bounds extrema
    non_zero_maxima = maxima[projection_profile[maxima] > 0]  # make sure that extrema are never in positions where only background is present

    return non_zero_maxima, smoothed_spline # TODO: remove spline


def convert_maxima_to_medial_seams(fully_connected_slice_maxima: List[List[Tuple]],
                                   slice_widths: List[int], image_height: int) -> numpy.ndarray:
    medial_seams = []
    for maxima_group in fully_connected_slice_maxima:
        medial_seam = []
        for i, (maximum, slice_width) in enumerate(zip(maxima_group[:-1], slice_widths)):
            slice_idx = maximum[1]
            next_maximum = maxima_group[i + 1]

            half_slice_width = round(slice_width / 2)
            x_start = sum(slice_widths[:slice_idx]) + half_slice_width + 1

            next_half_slice_width = round(slice_widths[slice_idx + 1] / 2)
            missing_slice_width = slice_width - half_slice_width
            x_end = x_start + missing_slice_width + next_half_slice_width

            y_coords, x_coords = line(maximum[0], x_start, next_maximum[0], x_end)
            medial_seam += list(zip(y_coords[:-1], x_coords[:-1]))

        # since we always draw lines from the middle of the slice we need to add padding for the first and last slice
        first_slice_idx = maxima_group[0][1]
        last_slice_idx = maxima_group[-1][1]
        first_slice_start = sum(slice_widths[:first_slice_idx])
        last_slice_end = sum(slice_widths[:last_slice_idx + 1])
        first_slice_half = [(medial_seam[0][0], x) for x in range(first_slice_start, medial_seam[0][1])]
        last_slice_half = [(medial_seam[-1][0], x) for x in range(medial_seam[-1][1] + 1, last_slice_end + 1)]
        adapted_medial_seam = first_slice_half + medial_seam + last_slice_half

        # pad all seams to the same lengths so that they can be correctly used with numpy
        padding_value = -2 * image_height  # set it to this value so that points are too far away to be accidentally used in distance calculation
        if first_slice_idx > 0:
            padding_left = [(padding_value, x) for x in range(first_slice_start)]
            adapted_medial_seam = padding_left + adapted_medial_seam
        if last_slice_idx < len(slice_widths) - 1:
            padding_right = [(padding_value, x) for x in range(last_slice_end, sum(slice_widths))]
            adapted_medial_seam = adapted_medial_seam + padding_right
        medial_seams.append(adapted_medial_seam)
    # TODO: off by one error in seam calc (missing x) => maybe just insert afterwards?
    assert all(len(m_s) == len(medial_seams[0]) for m_s in medial_seams), 'Medial seams are not of equal length'

    return numpy.asarray(medial_seams)


def calculate_medial_seams(image: Image.Image, r: int = 20, b: float = 0.0003) -> numpy.ndarray:
    """
    Args:
        image: Image to calculate medial seams for
        r: number of slices
        b: Smoothing parameter for csaps
    """
    # TODO function is quite long rn
    grayscale_image = image.convert("L")
    image_array = numpy.asarray(grayscale_image)
    sobel_image = ndimage.sobel(image_array)
    slices = numpy.array_split(sobel_image, r, axis=1)

    # Calculate maxima for each slice
    slice_maxima = []
    splines = []
    for image_slice in tqdm(slices, desc="Calculate seam maxima...", leave=False):
        maxima_locations, s = calculate_maxima_locations(image_slice, b)
        slice_maxima.append(maxima_locations)
        splines.append(s)  # TODO: remove

    # TODO: remove
    # from PIL import ImageDraw
    # image_draw = ImageDraw.Draw(image)
    # slice_widths = [s.shape[1] for s in slices]
    # for i, (slice, s) in tqdm(enumerate(zip(slice_maxima, splines)), desc="Processing slices"):
    #     x_start = sum(slice_widths[:i])
    #     x_end = sum(slice_widths[:i + 1])
    #     for m in tqdm(slice, desc="Drawing extrema"):
    #         if m < 0 or m > image.height:
    #             continue
    #         points = [(x_start, m), (x_end, m)]
    #         image_draw.line(points, fill=(0, 100, 255), width=5)
    #
    #     points = numpy.asarray([(s(x), x) for x in s.x], dtype=numpy.int32)
    #     min_p = points[:, 0].min()
    #     max_p = points[:, 0].max()
    #     blah = max_p - min_p
    #     scaled_points = (points * ((1 / blah) * slice_widths[i], 1)).astype(numpy.int32)
    #     new_min_p = scaled_points[:, 0].min()
    #     shifted_points = scaled_points + [x_start - new_min_p, 0]
    #
    #     points_to_draw = [(x,y) for x, y in shifted_points.tolist()]
    #     image_draw.line(points_to_draw, fill=(255, 0, 255), width=3)
    # image.show()

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

    # fully_connected_slice_maxima = [v for k, v in connected_slice_maxima.items() if len(v) == r]
    # TODO: rename ,
    #  2 is magic number
    fully_connected_slice_maxima = [v for k, v in connected_slice_maxima.items() if len(v) >= 2]

    slice_widths = [image_slice.shape[1] for image_slice in slices]
    medial_seams = convert_maxima_to_medial_seams(fully_connected_slice_maxima, slice_widths, image.height)

    return medial_seams


def get_dist_between_bbox_and_seam(bbox: LineBBox, seam: numpy.ndarray) -> int:
    bbox_x_mid = (bbox.left + bbox.right) // 2
    bbox_y_mid = (bbox.top + bbox.bottom) // 2
    dist = seam[bbox_x_mid, 0] - bbox_y_mid
    return dist


def map_bboxes_to_lines(bboxes: Tuple[LineBBox, ...], medial_seams: numpy.ndarray) -> Tuple[Dict[int, LineBBox], List[LineBBox]]:

    # Extract bounding boxes that intersect with at least one medial seam
    line_bbox_map = defaultdict(list)
    non_matched_bboxes = []
    for bbox in bboxes:
        # "relevant" means that matching x coordinates
        relevant_seam_fragments_y = medial_seams[:, bbox.left:bbox.right + 1, 0]
        # checks if any of the y coordinates of the seam fragments is within the bbox
        # (any/nonzero is just a way to select lines where at least one coordinate is within the bbox)
        seam_candidates = numpy.logical_and(bbox.top <= relevant_seam_fragments_y, bbox.bottom >= relevant_seam_fragments_y).any(axis=1).nonzero()[0].tolist()

        bbox.line_candidates = seam_candidates
        if len(seam_candidates) == 0:
            non_matched_bboxes.append(bbox)
        else:
            for seam_id in seam_candidates:
                line_bbox_map[seam_id].append(bbox)

    return line_bbox_map, non_matched_bboxes


def integrate_non_matched_bboxes(line_bbox_map, medial_seams, non_matched_bboxes):
    # See if non-matched bboxes can be matched based on context - mainly for small artifacts such as dots on the "i"
    addtional_line_bbox_map = defaultdict(list)
    for non_matched_bbox in non_matched_bboxes:
        # draw_bounding_boxes(image, (non_matched_bbox,), outline_color=(0, 0, 0))  # TODO: remove

        dists_to_seams = [(i, abs(get_dist_between_bbox_and_seam(non_matched_bbox, seam))) for i, seam in
                          enumerate(medial_seams)]
        closest_seam_id = min(dists_to_seams, key=lambda x: x[1])[0]

        seam_bboxes = line_bbox_map[closest_seam_id]
        # get closest bbox in line
        # TODO: take more than 1 on each side?
        dists_left = [(seam_bbox, non_matched_bbox.left - seam_bbox.right) for seam_bbox in seam_bboxes if
                      seam_bbox.right < non_matched_bbox.left]
        bbox_left = min(dists_left, key=lambda x: x[1], default=(None,))[0]
        dists_right = [(seam_bbox, seam_bbox.left - non_matched_bbox.right) for seam_bbox in seam_bboxes if
                       seam_bbox.left > non_matched_bbox.right]
        bbox_right = min(dists_right, key=lambda x: x[1], default=(None,))[0]

        if bbox_left is None and bbox_right is None:
            continue
        elif bbox_left is not None:
            neighboring_y_range = (bbox_left.top, bbox_left.bottom)
        elif bbox_right is not None:
            neighboring_y_range = (bbox_right.top, bbox_right.bottom)
        else:
            neighboring_y_range = (min(bbox_left.top, bbox_right.top), max(bbox_left.bottom, bbox_right.bottom))

        bbox_y_mid = (non_matched_bbox.top + non_matched_bbox.bottom) // 2
        if neighboring_y_range[0] <= bbox_y_mid <= neighboring_y_range[1]:
            addtional_line_bbox_map[closest_seam_id].append(non_matched_bbox)
    return addtional_line_bbox_map


def do_ranges_overlap(range1, range2) -> bool:
    # TODO: remove if not used again
    return range2[1] >= range1[0] and range1[1] >= range2[0]


def merge_line_bbox_maps(line_bbox_map, additional_line_bbox_map):
    new_line_bbox_map = defaultdict(list)
    for line_id, bboxes in line_bbox_map.items():
        new_line_bbox_map[line_id].extend(bboxes)
        if line_id in additional_line_bbox_map:
            new_line_bbox_map[line_id].extend(additional_line_bbox_map[line_id])
    return new_line_bbox_map


def split_lines(line_bbox_map: Dict[int, List[LineBBox]]) -> Dict[int, List[List[LineBBox]]]:
    """
    Split lines into multiple lines if spaces between bboxes (on the x-axis) are too large
    """
    partial_line_bbox_map = defaultdict(list)
    for line_id, line_bboxes in line_bbox_map.items():
        # calculate average bbox width in line
        sorted_line_boxes = sorted(line_bboxes, key=lambda bbox: bbox.left)
        avg_line_bbox_width = numpy.mean([bbox.width for bbox in sorted_line_boxes])
        current_line_part = []
        for i, line_bbox in enumerate(sorted_line_boxes):
            if i + 1 < len(sorted_line_boxes) and sorted_line_boxes[i + 1].left - line_bbox.right < 2 * avg_line_bbox_width:  # TODO: 2 is kinda a magic number
                current_line_part.append(line_bbox)
            else:
                # Space is too wide -> finish line segment and start new one
                current_line_part.append(line_bbox)
                partial_line_bbox_map[line_id].append(current_line_part)
                current_line_part = []
    return partial_line_bbox_map


def split_lines_into_clear_and_ambiguous(partial_line_bbox_map: Dict[int, List[List[LineBBox]]]) -> Tuple[Dict[int, List[List[LineBBox]]], Dict[int, List[List[LineBBox]]]]:
    clear_line_bbox_map = defaultdict(list)
    ambiguous_line_bbox_map = defaultdict(list)
    for line_id, line_parts in partial_line_bbox_map.items():
        for part in line_parts:
            is_ambiguous = any([b.is_ambiguous() for b in part])
            if is_ambiguous:
                ambiguous_line_bbox_map[line_id].append(part)
            else:
                clear_line_bbox_map[line_id].append(part)
    return ambiguous_line_bbox_map, clear_line_bbox_map


def main(args: argparse.Namespace):
    orig_image, orig_bboxes = load_image_and_bboxes(args)

    ### Preprocess image
    image, bboxes = preprocess(orig_image, orig_bboxes)
    # TODO: dynamically calculate this, maybe a (roughly) fixed slice width?
    #  r=5: slice_width=534
    #  r=6: slice_width=445 - seems to work best
    #  r=7: slice_width=381
    # r = 6
    r = 20
    # TODO: check if this works with double page layouts. If not, try to find a fix
    medial_seams = calculate_medial_seams(image, r=r)

    ### Visualization
    ####

    # Try to match all bboxes to medial seams
    # TODO: use again but maybe adapt
    line_bbox_map, non_matched_bboxes = map_bboxes_to_lines(bboxes, medial_seams)
    # TODO: try to bboxes that just slightly overlap line (e.g. "l" or "g")
    additional_line_bbox_map = integrate_non_matched_bboxes(line_bbox_map, medial_seams, non_matched_bboxes)

    new_line_bbox_map = merge_line_bbox_maps(line_bbox_map, additional_line_bbox_map)
    partial_line_bbox_map = split_lines(new_line_bbox_map)

    ambiguous_line_bbox_map, clear_line_bbox_map = split_lines_into_clear_and_ambiguous(partial_line_bbox_map)
    # TODO: is there any sorting of bboxes (based on x location)?

    # TODO: when saving, maybe split into clean lines (no preprocessing such as splitting or inserting) and postprocess lines

    ### Visualize lines
    # TODO: remove
    from PIL import ImageDraw
    image_draw = ImageDraw.Draw(image)
    for medial_seam in medial_seams:
        points = [(x, y) for y, x in medial_seam if y > 0]
        image_draw.line(points, fill=(0, 0, 255), width=5)
        # image_draw.line(points, fill=(255, 0, 0), width=5)

    # Only bboxes that were added later
    for line_id, line_bboxes in additional_line_bbox_map.items():
        draw_bounding_boxes(image, [BBox(b.left - 3, b.top - 3, b.right + 3, b.bottom + 3) for b in additional_line_bbox_map[line_id]], outline_color=(153, 0, 204))

    # All bboxes color-coded by line
    # for i, line_id in enumerate(partial_line_bbox_map.keys()):
    #     for part_id, line_part in enumerate(partial_line_bbox_map[line_id]):
    #         color = (255 if i % 2 == 0 else 0, 255 if part_id % 2 == 0 else 0, 0)
    #         draw_bounding_boxes(image, [b.bbox for b in line_part], outline_color=color)

    # clear boxes
    for i, line_id in enumerate(clear_line_bbox_map.keys()):
        for part_id, line_part in enumerate(partial_line_bbox_map[line_id]):
            color = (0, 255 if part_id % 2 == 0 else 100, 0)
            draw_bounding_boxes(image, [b.bbox for b in line_part], outline_color=color)

    # ambiguous boxes
    for i, line_id in enumerate(ambiguous_line_bbox_map.keys()):
        for part_id, line_part in enumerate(partial_line_bbox_map[line_id]):
            color = (255, 0, 0) if part_id % 2 == 0 else (250, 128, 114)
            draw_bounding_boxes(image, [b.bbox for b in line_part], outline_color=color)
    image.show()
    ####
    print()


if __name__ == '__main__':
    main(parse_args())
