from typing import List, Tuple

import numpy
from PIL import Image
from csaps import csaps
from scipy import ndimage
from skimage.draw import line
from tqdm import tqdm, trange

from stylegan_code_finder.handwriting_extraction.utils import LineBBox


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
    # make sure that extrema are never in positions where only background is present
    non_zero_maxima = maxima[projection_profile[maxima] > 0]

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


def calculate_medial_seams(image: Image.Image, r: int = 20, b: float = 0.0003,
                           min_num_maxima_in_seam: int = 2) -> numpy.ndarray:
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


