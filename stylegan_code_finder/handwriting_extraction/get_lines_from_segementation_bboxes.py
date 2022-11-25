import argparse
import copy
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy
from PIL import Image

from stylegan_code_finder.handwriting_extraction.medial_seam_calculation import get_dist_between_bbox_and_seam, \
    calculate_medial_seams, calculate_energy_map, calculate_separating_seams
from stylegan_code_finder.handwriting_extraction.utils import LineBBox, rotate_polygon, \
    radians_to_degrees, get_mutual_bbox, get_min_bbox, degrees_to_radians
from stylegan_code_finder.segmentation.evaluation.segmentation_visualization import draw_bounding_boxes
from stylegan_code_finder.utils.segmentation_utils import BBox

Line = List[LineBBox]
ProtoLine = List[Line]
LineMap = Dict[int, ProtoLine]


def set_opt_args_for_hw_extraction(parser):
    parser.add_argument('--slice-width', type=int, default=256,
                        help='Approximate width of slices, which will be used to calculate the number of slices used '
                             'for seam carving')
    parser.add_argument('--save-ambiguous-lines', action='store_true', default=False,
                        help='If set also saves images of ambiguous lines (these could be multiple lines that were '
                             'merged together')
    parser.add_argument('--min-num-bboxes', type=int, default=2,
                        help='Resulting lines with fewer bboxes will be discarded')
    parser.add_argument('--min-aspect-ratio', type=float, default=2., help='Minimum aspect ratio of a line')
    parser.add_argument('--min-line-area', type=int, default=10000,
                        help='Minimum area of a line')  # TODO: highly dependent on image size
    parser.add_argument('--debug', action='store_true', default=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=Path, help='Path to original image')
    parser.add_argument('segmented_image_path', type=Path, help='Path to segmented image')
    parser.add_argument('meta_info_path', type=Path, help='Path to meta info JSON')
    parser.add_argument('out_dir', type=Path, help='Path to root dir where resulting file should be saved')
    set_opt_args_for_hw_extraction(parser)
    return parser.parse_args()


def load_bbox_dict(meta_info_path: Path) -> Dict:
    with open(meta_info_path, 'r') as f:
        meta_information = json.load(f)
    return meta_information['bbox_dict']


def load_images(meta_info_path: Path, image_path: Path, segmented_image_path: Path) -> (Image.Image, Image.Image):
    with open(meta_info_path, 'r') as f:
        meta_information = json.load(f)
    segmentation_image = Image.open(segmented_image_path)
    original_image = Image.open(image_path)
    assert tuple(
        meta_information['image_size']) == segmentation_image.size, 'Image size does not match meta information'

    return original_image, segmentation_image


def get_bboxes_from_dict(bbox_dict):
    return tuple([LineBBox(*bbox) for bboxes in list(bbox_dict.values()) for bbox in bboxes])


def preprocess(segmentation_image: Image.Image, bboxes: Tuple[LineBBox, ...], padding: int = 10) -> (
        Image.Image, Line, List[int]):
    # Reduce segmentation image to area where bboxes are present. Slight performance speed up
    top_left = [max(min(values) - padding, 0) for values in list(zip(*bboxes))[:2]]
    bottom_right = [max(values) + 10 for values in list(zip(*bboxes))[2:]]
    min_segmentation_image = segmentation_image.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
    shifted_bboxes = numpy.asarray([b.bbox for b in bboxes]) - numpy.asarray([*top_left, *top_left])
    return min_segmentation_image, tuple((LineBBox(*bbox) for bbox in shifted_bboxes)), top_left


def map_bboxes_to_lines(bboxes: Tuple[LineBBox, ...], medial_seams: numpy.ndarray) -> Tuple[
    Dict[int, LineBBox], Line]:
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


def integrate_non_matched_bboxes(line_bbox_map: Dict[int, Line], medial_seams: numpy.ndarray, unmatched_bboxes: Line) -> \
        Tuple[Dict[int, Line], Line]:
    # See if non-matched bboxes can be matched based on context - mainly for small artifacts such as dots on the "i"
    addtional_line_bbox_map = defaultdict(list)
    still_unmatched_bboxes = []
    for unmatched_bbox in unmatched_bboxes:
        dists_to_seams = [(i, abs(get_dist_between_bbox_and_seam(unmatched_bbox, seam))) for i, seam in
                          enumerate(medial_seams)]
        closest_seam_id = min(dists_to_seams, key=lambda x: x[1])[0]

        seam_bboxes = line_bbox_map[closest_seam_id]
        # get closest bbox in line
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
            unmatched_bbox.line_candidates = [closest_seam_id]
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


def split_lines(line_bbox_map: Dict[int, Line], line_split_factor: float = 2) -> LineMap:
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


def postprocess_ambiguous_lines(ambiguous_line_bbox_map: LineMap, separating_seams: list) -> Tuple[LineMap, LineMap]:
    leftover_ambiguous_lines = defaultdict(list)
    additional_clear_lines = defaultdict(list)
    for line_id, ambiguous_line_parts in ambiguous_line_bbox_map.items():
        for line_part in ambiguous_line_parts:
            if len(line_part) == 1:
                continue
            unambig_parts = []
            current_unambig_part = []
            ambig_parts = []
            # TODO: comment this shit
            # TODO: debug this and maybe use separating seams
            for bbox in line_part:
                if bbox.is_ambiguous():
                    ambig_parts.append(bbox)
                    if len(current_unambig_part) > 0:
                        unambig_parts.append(current_unambig_part)
                        current_unambig_part = []
                else:
                    current_unambig_part.append(bbox)
            if len(current_unambig_part) > 0:
                unambig_parts.append(current_unambig_part)
            if len(unambig_parts) > 0:
                additional_clear_lines[line_id].append(unambig_parts)
            leftover_ambiguous_lines[line_id].append(ambig_parts)
    return additional_clear_lines, leftover_ambiguous_lines


def split_lines_into_clear_and_ambiguous(partial_line_bbox_map: LineMap, separating_seams: list) -> Tuple[LineMap, LineMap]:
    clear_line_bbox_map = defaultdict(list)
    ambiguous_line_bbox_map = defaultdict(list)
    for line_id, line_parts in partial_line_bbox_map.items():
        for part in line_parts:
            is_ambiguous = any([b.is_ambiguous() for b in part])
            if is_ambiguous:
                ambiguous_line_bbox_map[line_id].append(part)
            else:
                clear_line_bbox_map[line_id].append(part)

    # TODO: use separating seams for postprocessing
    additional_clear_line, leftover_ambiguous_line_map = postprocess_ambiguous_lines(ambiguous_line_bbox_map,
                                                                                     separating_seams)
    for line_id, line_parts in additional_clear_line.items():
        for line_part in line_parts:
            clear_line_bbox_map[line_id].extend(line_part)

    return leftover_ambiguous_line_map, clear_line_bbox_map


def format_line_bboxes(ambiguous_line_bbox_map: LineMap, clear_line_bbox_map: LineMap, shift: Tuple[int] = (0, 0)) -> List[Line]:
    merged_line_bbox_map = defaultdict(list)
    for d in (clear_line_bbox_map, ambiguous_line_bbox_map):
        for line_id, v in d.items():
            merged_line_bbox_map[line_id].extend(v)

    formatted_line_bboxes = []
    for line_id, line in dict(sorted(merged_line_bbox_map.items())).items():
        for part_id, line_part in enumerate(line):
            formatted_line = {'line_id': line_id, 'part_id': part_id, 'bboxes': [], 'line_candidates': []}
            is_ambiguous = False
            for line_bbox in line_part:
                line_bbox_dict = line_bbox.to_dict()
                # shift bbox back so it matches the original image
                shifted_bbox = tuple((numpy.asarray(line_bbox_dict['bbox']) + numpy.array((*shift, *shift))).tolist())
                formatted_line['bboxes'].append(shifted_bbox)
                if line_bbox.is_ambiguous():
                    is_ambiguous = True
                    formatted_line['line_candidates'].append(line_bbox.line_candidates)
                else:
                    formatted_line['line_candidates'].append(None)
            formatted_line['is_ambiguous'] = is_ambiguous
            formatted_line_bboxes.append(formatted_line)

    return formatted_line_bboxes


def visualize_bboxes(image, medial_seams, separating_seams, clear_line_bbox_map, additional_line_bbox_map,
                     ambiguous_line_bbox_map, partial_line_bbox_map, orig_bboxes):  # TODO: remove orig_bboxes
    from PIL import ImageDraw
    image_draw = ImageDraw.Draw(image)

    # Original bboxes
    draw_bounding_boxes(image, [bbox.bbox for bbox in orig_bboxes], outline_color=(238, 130, 238))

    for medial_seam in medial_seams:
        points = [(x, y) for y, x in medial_seam if y > 0]
        image_draw.line(points, fill=(0, 0, 255), width=5)
        # image_draw.line(points, fill=(255, 0, 0), width=5)
    for separating_seam in separating_seams:
        points = [(x, y) for y, x in separating_seam]
        image_draw.line(points, fill='purple', width=5)
    # Only bboxes that were added later
    for line_id, line_bboxes in additional_line_bbox_map.items():
        draw_bounding_boxes(image, [BBox(b.left - 3, b.top - 3, b.right + 3, b.bottom + 3) for b in
                                    additional_line_bbox_map[line_id]], outline_color=(153, 0, 204))
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
    # for i, line_id in enumerate(ambiguous_line_bbox_map.keys()):
    #     for part_id, line_part in enumerate(partial_line_bbox_map[line_id]):
    #         color = (255, 0, 0) if part_id % 2 == 0 else (250, 128, 114)
    #         draw_bounding_boxes(image, [b.bbox for b in line_part], outline_color=color)
    image.show()
    # TODO: clean
    # debug_image_path = Path('test/extracted_lines/debug.png')
    # image.save(debug_image_path)


def extract_lines_from_image(orig_bboxes: Tuple[LineBBox, ...], orig_segmented_image: Image.Image,
                             slice_width: int = 256, b: float = 0.0003, min_num_maxima_in_seam: int = 2,
                             debug: bool = False) -> dict:
    ### Preprocess image
    # TODO maybe remove this "optimization" completely
    # segmented_image, bbox, bbox_shift = preprocess(orig_segmented_image, orig_bboxes)
    segmented_image = orig_segmented_image
    bbox = orig_bboxes
    r = segmented_image.width // slice_width
    logging.info(f'Slice width for r={r}: {segmented_image.width // r}')

    medial_seams = calculate_medial_seams(segmented_image, r=r, b=b, min_num_maxima_in_seam=min_num_maxima_in_seam)
    sigma = 3.0  # TODO: to param
    energy_map = calculate_energy_map(segmented_image, sigma=sigma)
    separating_seams = calculate_separating_seams(medial_seams, energy_map)
    if len(medial_seams) > 0:
        # Try to match all bboxes to medial seams
        line_bbox_map, unmatched_bboxes = map_bboxes_to_lines(bbox, medial_seams)
        additional_line_bbox_map, still_unmatched_bboxes = integrate_non_matched_bboxes(line_bbox_map, medial_seams,
                                                                                        unmatched_bboxes)
        new_line_bbox_map = merge_line_bbox_maps(line_bbox_map, additional_line_bbox_map)
        partial_line_bbox_map = split_lines(new_line_bbox_map)
        ambiguous_line_bbox_map, clear_line_bbox_map = split_lines_into_clear_and_ambiguous(partial_line_bbox_map,
                                                                                            separating_seams)

        if debug:
            # TODO: have another look at visualization, seems fishy (the drawing of ambiguous boxes)
            visualize_bboxes(segmented_image, medial_seams, separating_seams, clear_line_bbox_map,
                             additional_line_bbox_map, ambiguous_line_bbox_map, partial_line_bbox_map, orig_bboxes)

        # TODO: postprocess ambig. lines - maybe use original seam carving code for this one
        # final_bboxes = format_line_bboxes(ambiguous_line_bbox_map, clear_line_bbox_map, bbox_shift)
        final_bboxes = format_line_bboxes(ambiguous_line_bbox_map, clear_line_bbox_map)
    else:
        final_bboxes = []

    results = {
        'seam_carving_hyperparams': {
            'slice_width': slice_width,
            'r': r,
            'b': b,
            'min_num_maxima_in_seam': min_num_maxima_in_seam
        },
        'lines': final_bboxes
    }
    return results


def crop_min_bbox_from_image(min_bbox, mid_point, rotation, image):
    # TODO: clean whole function
    crop_bbox = (min_bbox[:, 0].min(), min_bbox[:, 1].min(), min_bbox[:, 0].max(), min_bbox[:, 1].max())
    small_image = image.crop(crop_bbox)
    # TODO: to coreect rotation was a good idea, but would need some more thinking. Especially rotations close to
    #  90 degrees cause problems. These bboxes are not really rotated 88° but more like -2°
    if False and rotation != 0.0:
        # box_line_points = tuple(min_bbox[:2].tolist())
        # rad_rotation = get_line_rotation(box_line_points)  # TODO: remove function too
        rad_rotation = degrees_to_radians(rotation)

        # mid_point = numpy.mean(min_bbox, axis=0, dtype=numpy.int32)
        rot = rotate_polygon(numpy.array(min_bbox), numpy.array(mid_point), rad_rotation)  # TODO: does not seems to work correctly for larger tilts
        # TODO: not sure that points are ordered like this
        rotated_min_bbox = BBox(rot[0][0], rot[0][1], rot[2][0], rot[2][1])  # TODO: could also average the other points or maybe add an assertion that rotation is not too high
        # shifted_min_bbox = BBox(rotated_min_bbox.left - crop_bbox[0], rotated_min_bbox.top - crop_bbox[1],
        #                         rotated_min_bbox.right - crop_bbox[0], rotated_min_bbox.bottom - crop_bbox[1])

        small_image = small_image.rotate(-radians_to_degrees(rad_rotation), expand=False)
    else:
        rotated_min_bbox = BBox(min(min_bbox[:, 0]), min(min_bbox[:, 1]), max(min_bbox[:, 0]), max(min_bbox[:, 1]))
    shifted_min_bbox = BBox(rotated_min_bbox.left - crop_bbox[0], rotated_min_bbox.top - crop_bbox[1],
                            rotated_min_bbox.right - crop_bbox[0], rotated_min_bbox.bottom - crop_bbox[1])

    assert (
            shifted_min_bbox.left >= 0 and
            shifted_min_bbox.top >= 0 and
            shifted_min_bbox.right >= 0 and
            shifted_min_bbox.bottom >= 0 and
            shifted_min_bbox.left <= small_image.width and
            shifted_min_bbox.top <= small_image.height and
            shifted_min_bbox.right <= small_image.width and
            shifted_min_bbox.bottom <= small_image.height
    ), 'Shifted bboxes was calculated improperly and is overflowing'
    line_image = small_image.crop(shifted_min_bbox)

    return line_image


def crop_lines(image: Image.Image, line_bboxes: dict, original_image_path, min_num_bboxes: int = 2,  # TODO: maybe remove original_image_path
               min_aspect_ratio: float = 2., min_line_area: int = 10000, keep_ambiguous_lines: bool = False):
    num_cropped_images = 0
    for line_infos in line_bboxes:
        # Filter ambiguous lines or noise, such as lines consisting of only one bboxes, small bboxes, or malformed
        # lines (lines that have a small aspect ratio are unlikely to consist of multiple words)
        new_bbox = get_mutual_bbox(line_infos['bboxes']).bbox
        if (
                (not keep_ambiguous_lines and line_infos['is_ambiguous']) or
                (len(line_infos['bboxes']) < min_num_bboxes or
                new_bbox.width / new_bbox.height < min_aspect_ratio or
                new_bbox.width * new_bbox.height < min_line_area)
        ):
            line_infos['filtered_out'] = True
            continue

        min_bbox, mid_point, rotation = get_min_bbox(line_infos)
        line_image = crop_min_bbox_from_image(min_bbox, mid_point, rotation, image)
        line_infos['image'] = line_image
        line_infos['filtered_out'] = False
        num_cropped_images += 1

    if num_cropped_images == 0:
        logging.warning(f'{original_image_path}: No lines were cropped since all line candidates were filtered out.')

    return line_bboxes


def save_lines(line_bboxes: List[dict], original_image_path, out_dir: Path):
    for line_infos in line_bboxes:
        if line_infos['filtered_out']:
            continue
        filename = f"{original_image_path.with_suffix('')}__{line_infos['line_id']}_{line_infos['part_id']}_{'ambiguous' if line_infos['is_ambiguous'] else 'clean'}.png"
        out_path = out_dir / filename
        line_infos['image'].save(out_path)


def process_segmentation(bbox_dict: dict, segmented_image: Image.Image, original_image: Image.Image, slice_width: int,
                         original_image_name: Path, out_dir: Optional[Path] = None, keep_ambiguous_lines: bool = False,
                         min_num_bboxes: int = 2, min_aspect_ratio: float = 2., min_line_area: int = 10000,
                         debug: bool = False):  # TODO: specify return
    bboxes = get_bboxes_from_dict(bbox_dict)
    results = extract_lines_from_image(bboxes, segmented_image, slice_width=slice_width, debug=debug)

    if len(results['lines']) > 0:
        results['line_hyperparams'] = {
            'min_num_bboxes': min_num_bboxes,
            'min_aspect_ratio': min_aspect_ratio,
            'min_line_area': min_line_area
        }

        results['lines'] = crop_lines(original_image, results['lines'], original_image_name,
                                      min_num_bboxes=min_num_bboxes, min_aspect_ratio=min_aspect_ratio,
                                      min_line_area=min_line_area, keep_ambiguous_lines=keep_ambiguous_lines)
    else:
        logging.warning(f'{original_image_name}: No medial seams found, so no bboxes will be extracted.')

    if out_dir is not None:
        assert out_dir is not None, 'Output directory must be specified if save_outputs is True.'
        out_dir.mkdir(parents=True, exist_ok=True)
        save_lines(results['lines'], original_image_name, out_dir)

        out_path = out_dir / original_image_name.with_suffix('.json')
        results_without_images = copy.deepcopy(results)
        for line_infos in results_without_images['lines']:
            line_infos.pop('image', None)

        with open(out_path, 'w') as f:
            json.dump(results_without_images, f, indent=2)

    return results


def main(args: argparse.Namespace):
    orig_image, segmented_image = load_images(args.meta_info_path, args.image_path, args.segmented_image_path)
    bbox_dict = load_bbox_dict(args.meta_info_path)
    results_info = process_segmentation(bbox_dict=bbox_dict, segmented_image=segmented_image, original_image=orig_image,
                                        slice_width=args.slice_width, original_image_name=Path(args.image_path.name),
                                        out_dir=args.out_dir, keep_ambiguous_lines=args.save_ambiguous_lines,
                                        min_num_bboxes=args.min_num_bboxes, min_aspect_ratio=args.min_aspect_ratio,
                                        min_line_area=args.min_line_area, debug=args.debug)


if __name__ == '__main__':
    main(parse_args())
