from typing import List, Tuple, Optional

import numpy
from PIL import Image
from csaps import csaps
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.draw import line
from tqdm import tqdm, trange

from stylegan_code_finder.handwriting_extraction.utils import LineBBox


# Code adapted from: https://github.com/hendraet/seam-carving


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
    fully_connected_slice_maxima = sorted(fully_connected_slice_maxima, key=lambda x: min([el[0] for el in x]))

    slice_widths = [image_slice.shape[1] for image_slice in slices]
    medial_seams = convert_maxima_to_medial_seams(fully_connected_slice_maxima, slice_widths, image.size)

    return medial_seams


def get_dist_between_bbox_and_seam(bbox: LineBBox, seam: numpy.ndarray) -> int:
    bbox_x_mid = (bbox.left + bbox.right) // 2
    bbox_y_mid = (bbox.top + bbox.bottom) // 2
    dist = seam[bbox_x_mid, 0] - bbox_y_mid
    return dist


def calculate_energy_map(original_image: Image.Image, sigma: float = 3.0) -> numpy.ndarray:
    grayscale_image = original_image.copy().convert("L")
    smoothed_image = gaussian_filter(numpy.asarray(grayscale_image), sigma=sigma, output=numpy.float64)

    # calculate the two addends of the formula independently but in a fast way
    # the arrays have to be padded with 2 rows/column because they are shifted by 1 and need additional padding to
    # handle the calculation at the edges, e.g. that the value at pixel_value[j - 1] = 0 if j - 1 < 0
    vertical_padding = numpy.zeros((grayscale_image.height, 2))
    j_plus_one = numpy.concatenate((smoothed_image, vertical_padding), axis=1)
    j_minus_one = numpy.concatenate((vertical_padding, smoothed_image), axis=1)
    vertical_energy_map = numpy.absolute((j_plus_one - j_minus_one) / 2)[:, 1:-1]

    horizontal_padding = numpy.zeros((2, grayscale_image.width))
    i_plus_one = numpy.concatenate((smoothed_image, horizontal_padding), axis=0)
    i_minus_one = numpy.concatenate((horizontal_padding, smoothed_image), axis=0)
    horizontal_energy_map = numpy.absolute((i_plus_one - i_minus_one) / 2)[1:-1, :]
    energy_map = vertical_energy_map + horizontal_energy_map

    # The calculation at the edges produces misleading results since one of the addends is always 0. Therefore, we set
    # these regions to 0, which corresponds to the energy of background pixels.
    energy_map[0, :] = 0
    energy_map[-1, :] = 0
    energy_map[:, 0] = 0
    energy_map[:, -1] = 0

    return energy_map


def get_local_energy_map(medial_seam: numpy.ndarray, next_medial_seam: numpy.ndarray,
                         energy_map: numpy.ndarray) -> Tuple[Optional[numpy.ndarray], Optional[Tuple[int, int]]]:
    # check if partial medial seams are overlapping and a proper energy map can be calculated
    overlap = numpy.intersect1d(medial_seam[:, 1], next_medial_seam[:, 1])
    if len(overlap) == 0:
        return None, None

    x_start = overlap.min()
    x_end = overlap.max()
    width = x_end + 1 - x_start

    reduced_medial_seam = medial_seam[numpy.logical_and(
        medial_seam[:, 1] >= x_start,
        medial_seam[:, 1] <= x_end
    )]
    reduced_next_medial_seam = next_medial_seam[numpy.logical_and(
        next_medial_seam[:, 1] >= x_start,
        next_medial_seam[:, 1] <= x_end
    )]
    assert reduced_next_medial_seam.shape == reduced_medial_seam.shape

    i_start = reduced_medial_seam[:, 0].min()
    absolute_height_diff = reduced_next_medial_seam[:, 0].max() - i_start
    assert absolute_height_diff >= 0, 'Given seams do not seem to be orderd, implying preprocessing went wrong.'
    i_end = i_start + absolute_height_diff
    real_local_energy_map = energy_map[i_start:i_end + 1, x_start:x_end + 1].copy()

    # The points between two medial seems rarely form a 2D array. Thus, we create a 2D array that contains all relevant
    # values from the energy map and set all irrelevant pixel to the sum of all values in the energy map (called
    # "empty value"). Since we try to minimise a sum of pixels and values in the energy are always > 0 this
    # should be sufficient so that those "empty values" are never part of an actual solution.
    empty_value = numpy.sum(real_local_energy_map)
    mask = numpy.linspace([i_start] * width, [i_end] * width, num=absolute_height_diff + 1)
    mask = numpy.where(
        numpy.logical_or(
            mask < reduced_medial_seam[:, 0],
            mask > reduced_next_medial_seam[:, 0]
        ), False, True)

    assert real_local_energy_map.shape == mask.shape
    local_energy_map = numpy.where(mask, real_local_energy_map, empty_value)

    return local_energy_map, (x_start, x_end)


def calculate_minimum_energy_map(medial_seam: numpy.ndarray, local_energy_map: numpy.ndarray, x_range: Tuple[int, int]) -> numpy.ndarray:
    # use DP to calculate the optimal energy paths for each starting pixel
    width = x_range[-1] + 1 - x_range[0]
    height = local_energy_map.shape[0]
    minimum_energy_map = numpy.zeros_like(local_energy_map)
    minimum_energy_map[:, 0] = local_energy_map[:, 0]

    for j in trange(1, width, desc="Calculating minimum energy map...", leave=False):
        for i in range(height):
            previous_energies = [minimum_energy_map[i, j - 1]]
            if i - 1 >= 0:
                previous_energies += [minimum_energy_map[i - 1, j - 1]]
            if i + 1 < height:
                previous_energies += [minimum_energy_map[i + 1, j - 1]]
            minimum_energy_map[i, j] = local_energy_map[i, j] + min(previous_energies)

    return minimum_energy_map


def get_optimal_separating_seam(medial_seam: numpy.ndarray, local_energy_map: numpy.ndarray,
                                minimum_energy_map: numpy.ndarray, x_range: int) -> List[Tuple]:
    width = x_range[-1] + 1 - x_range[0]
    height = local_energy_map.shape[0]
    y_start = medial_seam[:, 0].min()
    current_y = numpy.argmin(minimum_energy_map[:, -1])
    optimal_seam = [(current_y, width - 1)]

    for x in reversed(range(0, width - 1)):
        best_y = current_y
        best_energy = minimum_energy_map[current_y, x]
        if current_y - 1 >= 0 and minimum_energy_map[current_y - 1, x] < best_energy:
            best_y = current_y - 1
            best_energy = minimum_energy_map[best_y, x]
        if current_y + 1 < height and minimum_energy_map[current_y + 1, x] < best_energy:
            best_y = current_y + 1
        optimal_seam.append((best_y, x))
        current_y = best_y

    absolute_optimal_seam = [(y + y_start, x + x_range[0]) for y, x in reversed(optimal_seam)]
    return absolute_optimal_seam


def calculate_separating_seams(medial_seams: numpy.ndarray, energy_map: numpy.ndarray) -> List:
    separating_seams = []
    for seam_idx, medial_seam in enumerate(tqdm(medial_seams[:-1], desc="Calculating separating seams...",
                                                leave=False)):
        next_medial_seam = medial_seams[seam_idx + 1]
        # Since seams can be partial, we shorten the ndarray to the range of x values that have meaningful y-values.
        reduced_medial_seam = medial_seam[medial_seam[:, 0] > 0]
        reduced_next_medial_seam = next_medial_seam[next_medial_seam[:, 0] > 0]

        # TODO: technically two neighboring seams do not have to be relevant for each other, e.g., if they are not
        #  overlapping or are too far from each other
        local_energy_map, x_range = get_local_energy_map(reduced_medial_seam, reduced_next_medial_seam, energy_map)
        if local_energy_map is None:
            continue
        minimum_energy_map = calculate_minimum_energy_map(reduced_medial_seam, local_energy_map, x_range)
        optimal_separating_seam = get_optimal_separating_seam(reduced_medial_seam, local_energy_map,
                                                              minimum_energy_map, x_range)
        separating_seams.append(optimal_separating_seam)

    return separating_seams
