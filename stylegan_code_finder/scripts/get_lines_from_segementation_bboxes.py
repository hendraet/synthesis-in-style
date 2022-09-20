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

from scripts.script_utils import LineBBox
from segmentation.evaluation.segmentation_visualization import draw_bounding_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=Path, help='Path to image')
    parser.add_argument('meta_info_path', type=Path, help='Path to meta info JSON')
    parser.add_argument('--slice-width', type=int, default=256,
                        help='Approximate width of slices, which will be used to calculate the number of slices used '
                             'for seam carving')
    parser.add_argument('--debug', action='store_true', default=False)
    return parser.parse_args()


def load_image_and_bboxes(meta_info_path: Path, image_path: Path) -> (Image.Image, Tuple[LineBBox, ...]):
    with open(meta_info_path, 'r') as f:
        meta_information = json.load(f)
    segmentation_image = Image.open(image_path)
    assert tuple(meta_information['image_size']) == segmentation_image.size, 'Image size does not match meta information'

    bbox_dict = meta_information['bbox_dict']
    bboxes = tuple([LineBBox(*bbox) for bboxes in list(bbox_dict.values()) for bbox in bboxes])
    return segmentation_image, bboxes


def preprocess(segmentation_image: Image.Image, bboxes: Tuple[LineBBox, ...], padding: int = 10) -> (
Image.Image, List[LineBBox]):
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
    non_zero_maxima = maxima[projection_profile[
                                 maxima] > 0]  # make sure that extrema are never in positions where only background is present

    return non_zero_maxima


def convert_maxima_to_medial_seams(fully_connected_slice_maxima: List[List[Tuple]],
                                   slice_widths: List[int], image_size: Tuple[int, int]) -> numpy.ndarray:
    slice_mids = numpy.cumsum(slice_widths) - numpy.asarray(slice_widths) // 2
    medial_seams = []
    for j, maxima_group in enumerate(fully_connected_slice_maxima):
        medial_seam = []
        for i, (maximum, slice_width) in enumerate(zip(maxima_group[:-1], slice_widths)):
            slice_idx = maximum[1]
            next_maximum = maxima_group[i + 1]
            x_start = slice_mids[slice_idx] + 1
            x_end = slice_mids[slice_idx + 1]
            y_coords, x_coords = line(maximum[0], x_start, next_maximum[0], x_end)
            points = list(zip(y_coords, x_coords))
            if len(x_coords) > x_end - x_start + 1:
                seen_x_coords = set()
                # Only happens when difference between y coordinates of maxima is larger than slice width
                for p in points:
                    if p[1] not in seen_x_coords:
                        medial_seam.append(p)
                        seen_x_coords.add(p[1])
            else:
                medial_seam += points

        # since we always draw lines from the middle of the slice we need to add padding for the first and last slice
        first_slice_idx = maxima_group[0][1]
        last_slice_idx = maxima_group[-1][1]
        first_slice_start = sum(slice_widths[:first_slice_idx])
        last_slice_end = sum(slice_widths[:last_slice_idx + 1])
        first_slice_half = [(medial_seam[0][0], x) for x in range(first_slice_start, medial_seam[0][1])]
        last_slice_half = [(medial_seam[-1][0], x) for x in range(medial_seam[-1][1] + 1, last_slice_end)]
        adapted_medial_seam = first_slice_half + medial_seam + last_slice_half

        # pad all seams to the same lengths so that they can be correctly used with numpy
        padding_value = -2 * image_size[
            1]  # set it to this value so that points are too far away to be accidentally used in distance calculation
        if first_slice_idx > 0:
            padding_left = [(padding_value, x) for x in range(first_slice_start)]
            adapted_medial_seam = padding_left + adapted_medial_seam
        if last_slice_idx < len(slice_widths) - 1:
            padding_right = [(padding_value, x) for x in range(last_slice_end, sum(slice_widths))]
            adapted_medial_seam = adapted_medial_seam + padding_right
        assert (len(adapted_medial_seam) == image_size[0])
        medial_seams.append(adapted_medial_seam)
    return numpy.asarray(medial_seams)


def calculate_medial_seams(image: Image.Image, r: int = 20, b: float = 0.0003, min_num_maxima_in_seam: int = 2) -> numpy.ndarray:
    """
    Args:
        image: Image to calculate medial seams for
        r: number of slices
        b: Smoothing parameter for csaps
        min_num_maxima_in_seam: Minimum number of maxima that need to be present in a seam for it to be considered valid
    """
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

    fully_connected_slice_maxima = [v for k, v in connected_slice_maxima.items() if len(v) >= min_num_maxima_in_seam]

    slice_widths = [image_slice.shape[1] for image_slice in slices]
    medial_seams = convert_maxima_to_medial_seams(fully_connected_slice_maxima, slice_widths, image.size)

    return medial_seams


def get_dist_between_bbox_and_seam(bbox: LineBBox, seam: numpy.ndarray) -> int:
    bbox_x_mid = (bbox.left + bbox.right) // 2
    bbox_y_mid = (bbox.top + bbox.bottom) // 2
    dist = seam[bbox_x_mid, 0] - bbox_y_mid
    return dist


def map_bboxes_to_lines(bboxes: Tuple[LineBBox, ...], medial_seams: numpy.ndarray) -> Tuple[
    Dict[int, LineBBox], List[LineBBox]]:
    # Extract bounding boxes that intersect with at least one medial seam
    line_bbox_map = defaultdict(list)
    non_matched_bboxes = []
    for bbox in bboxes:
        # "relevant" means that matching x coordinates
        relevant_seam_fragments_y = medial_seams[:, bbox.left:bbox.right + 1, 0]
        # checks if any of the y coordinates of the seam fragments is within the bbox
        # (any/nonzero is just a way to select lines where at least one coordinate is within the bbox)
        seam_candidates = \
        numpy.logical_and(bbox.top <= relevant_seam_fragments_y, bbox.bottom >= relevant_seam_fragments_y).any(
            axis=1).nonzero()[0].tolist()

        bbox.line_candidates = seam_candidates
        if len(seam_candidates) == 0:
            non_matched_bboxes.append(bbox)
        else:
            for seam_id in seam_candidates:
                line_bbox_map[seam_id].append(bbox)

    return line_bbox_map, non_matched_bboxes


def integrate_non_matched_bboxes(line_bbox_map: Dict[int, List[LineBBox]], medial_seams: numpy.ndarray, unmatched_bboxes: List[LineBBox]) -> Tuple[Dict[int, List[LineBBox]], List[LineBBox]]:
    # See if non-matched bboxes can be matched based on context - mainly for small artifacts such as dots on the "i"
    addtional_line_bbox_map = defaultdict(list)
    still_unmatched_bboxes = []
    for unmatched_bbox in unmatched_bboxes:
        dists_to_seams = [(i, abs(get_dist_between_bbox_and_seam(unmatched_bbox, seam))) for i, seam in
                          enumerate(medial_seams)]
        closest_seam_id = min(dists_to_seams, key=lambda x: x[1])[0]

        seam_bboxes = line_bbox_map[closest_seam_id]
        # get closest bbox in line
        # TODO: take more than 1 on each side?
        dists_left = [(seam_bbox, unmatched_bbox.left - seam_bbox.right) for seam_bbox in seam_bboxes if
                      seam_bbox.right < unmatched_bbox.left]
        bbox_left = min(dists_left, key=lambda x: x[1], default=(None,))[0]
        dists_right = [(seam_bbox, seam_bbox.left - unmatched_bbox.right) for seam_bbox in seam_bboxes if
                       seam_bbox.left > unmatched_bbox.right]
        bbox_right = min(dists_right, key=lambda x: x[1], default=(None,))[0]

        if bbox_left is None and bbox_right is None:
            continue
        elif bbox_left is not None:
            neighboring_y_range = (bbox_left.top, bbox_left.bottom)
        elif bbox_right is not None:
            neighboring_y_range = (bbox_right.top, bbox_right.bottom)
        else:
            neighboring_y_range = (min(bbox_left.top, bbox_right.top), max(bbox_left.bottom, bbox_right.bottom))

        bbox_y_mid = (unmatched_bbox.top + unmatched_bbox.bottom) // 2
        if neighboring_y_range[0] <= bbox_y_mid <= neighboring_y_range[1]:
            addtional_line_bbox_map[closest_seam_id].append(unmatched_bbox)
        else:
            still_unmatched_bboxes.append(unmatched_bbox)
    return addtional_line_bbox_map, still_unmatched_bboxes


def merge_line_bbox_maps(line_bbox_map, additional_line_bbox_map):
    new_line_bbox_map = defaultdict(list)
    for line_id, bboxes in line_bbox_map.items():
        new_line_bbox_map[line_id].extend(bboxes)
        if line_id in additional_line_bbox_map:
            new_line_bbox_map[line_id].extend(additional_line_bbox_map[line_id])
    return new_line_bbox_map


def split_lines(line_bbox_map: Dict[int, List[LineBBox]], line_split_factor: float = 2) -> Dict[int, List[List[LineBBox]]]:
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
            if (
                    i + 1 < len(sorted_line_boxes) and
                    sorted_line_boxes[i + 1].left - line_bbox.right < line_split_factor * avg_line_bbox_width
            ):
                current_line_part.append(line_bbox)
            else:
                # Space is too wide -> finish line segment and start new one
                current_line_part.append(line_bbox)
                partial_line_bbox_map[line_id].append(current_line_part)
                current_line_part = []
    return partial_line_bbox_map


def split_lines_into_clear_and_ambiguous(partial_line_bbox_map: Dict[int, List[List[LineBBox]]]) -> Tuple[
    Dict[int, List[List[LineBBox]]], Dict[int, List[List[LineBBox]]]]:
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


def format_line_bboxes(ambiguous_line_bbox_map, clear_line_bbox_map):
    # TODO: annotations needed - maybe used annotation class for maps
    merged_line_bbox_map = defaultdict(list)
    for d in (clear_line_bbox_map, ambiguous_line_bbox_map):
        for line_id, v in d.items():
            merged_line_bbox_map[line_id].extend(v)

    formatted_line_bboxes = []
    for line_id, line in dict(sorted(merged_line_bbox_map.items())).items():
        for part_id, line_part in enumerate(line):
            if len(line_part) == 1:
                continue  # TODO: remove or maybe move somewhere else (probably to a separate "filter noise" function)
            formatted_line = {'line_id': line_id, 'part_id': part_id, 'bboxes': [], 'line_candidates': []}
            is_ambiguous = False
            for line_bbox in line_part:
                line_bbox_dict = line_bbox.to_dict()
                formatted_line['bboxes'].append(line_bbox_dict['bbox'])
                if line_bbox.is_ambiguous():
                    is_ambiguous = True
                    formatted_line['line_candidates'].append(line_bbox.line_candidates)
                else:
                    formatted_line['line_candidates'].append(None)
            formatted_line['is_ambiguous'] = is_ambiguous
            formatted_line_bboxes.append(formatted_line)

    return formatted_line_bboxes


def visualize_bboxes(image, medial_seams, clear_line_bbox_map, additional_line_bbox_map, ambiguous_line_bbox_map,
                     partial_line_bbox_map):
    from PIL import ImageDraw
    image_draw = ImageDraw.Draw(image)
    for medial_seam in medial_seams:
        points = [(x, y) for y, x in medial_seam if y > 0]
        image_draw.line(points, fill=(0, 0, 255), width=5)
        # image_draw.line(points, fill=(255, 0, 0), width=5)
    # Only bboxes that were added later
    # for line_id, line_bboxes in additional_line_bbox_map.items():
    #     draw_bounding_boxes(image, [BBox(b.left - 3, b.top - 3, b.right + 3, b.bottom + 3) for b in
    #                                 additional_line_bbox_map[line_id]], outline_color=(153, 0, 204))
    # All bboxes color-coded by line
    for i, line_id in enumerate(partial_line_bbox_map.keys()):
        for part_id, line_part in enumerate(partial_line_bbox_map[line_id]):
            color = (255 if i % 2 == 0 else 0, 255 if part_id % 2 == 0 else 0, 0)
            draw_bounding_boxes(image, [b.bbox for b in line_part], outline_color=color)
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
    # image.show()  # TODO: re-comment


def get_mutual_bbox(bboxes):
    l, t, r, b = zip(*bboxes)
    return LineBBox(min(l), min(t), max(r), max(b))


def crop_and_save_lines(image, line_bboxes, save_ambiguous_lines=False):
    # TODO: assert that image may be original image
    bboxes = []
    for line_infos in line_bboxes:
        if not save_ambiguous_lines and line_infos['is_ambiguous']:
            continue
        new_bbox = get_mutual_bbox(line_infos['bboxes']).bbox
        if new_bbox.width / new_bbox.height < 2.:  # TODO: maybe use this to filter noise, but has to be moved somewhere else
            continue
        bboxes.append(new_bbox)
    draw_bounding_boxes(image, bboxes, outline_color=(93, 63, 211))  # TODO: remove
    image.show()

    print()


def extract_lines_from_image(orig_bboxes: Tuple[LineBBox, ...], orig_image: Image.Image, slice_width: int = 256,
                             save_ambiguous_lines: bool = False, b: float = 0.0003, min_num_maxima_in_seam: int = 2,
                             debug: bool = False):
    # TODO: proper noise removal in any way (i.e. multiple small bboxes)
    ### Preprocess image
    image, bbox = preprocess(orig_image, orig_bboxes)
    r = image.width // slice_width
    print(f'Slice width for r={r}: {image.width // r}')

    medial_seams = calculate_medial_seams(image, r=r, b=b, min_num_maxima_in_seam=min_num_maxima_in_seam)

    # Try to match all bboxes to medial seams
    line_bbox_map, unmatched_bboxes = map_bboxes_to_lines(bbox, medial_seams)
    additional_line_bbox_map, still_unmatched_bboxes = integrate_non_matched_bboxes(line_bbox_map, medial_seams,
                                                                                    unmatched_bboxes)
    new_line_bbox_map = merge_line_bbox_maps(line_bbox_map, additional_line_bbox_map)
    partial_line_bbox_map = split_lines(new_line_bbox_map)
    # TODO: check if one could filter single overlapping bboxes out, e.g., if only last box in line overlaps, just remove this box and save rest as non_ambiguous line
    ambiguous_line_bbox_map, clear_line_bbox_map = split_lines_into_clear_and_ambiguous(partial_line_bbox_map)

    if debug:
        visualize_bboxes(image, medial_seams, clear_line_bbox_map, additional_line_bbox_map, ambiguous_line_bbox_map,
                         partial_line_bbox_map)

    # TODO: when saving, maybe split into clean lines (no preprocessing such as splitting or inserting) and postprocess lines
    #  - maybe use original seam carving code for this one

    # TODO: actually extract lines and save them as image together with bbox infos (and if they are ambiguous or not)
    final_bboxes = format_line_bboxes(ambiguous_line_bbox_map, clear_line_bbox_map)
    results = {
        'seam_carving_hyperparams': {
            'slice_width': slice_width,
            'r': r,
            'b': b,
            'min_num_maxima_in_seam': min_num_maxima_in_seam
        },
        'lines': final_bboxes
    }
    crop_and_save_lines(image, final_bboxes, save_ambiguous_lines=save_ambiguous_lines)  # TODO maybe only extract and return
    return results


def process_image(bboxes, image, slice_width, debug=False):
    # TODO: make sure that image is segmented image when used in extract_hw script
    results = extract_lines_from_image(bboxes, image, slice_width=slice_width, debug=debug)  # TODO: set parameter for ambig lines via argparse
    return results


def main(args: argparse.Namespace):
    orig_image, orig_bboxes = load_image_and_bboxes(args.meta_info_path, args.image_path)
    process_image(orig_bboxes, orig_image, slice_width=args.slice_width, debug=args.debug)


if __name__ == '__main__':
    main(parse_args())
