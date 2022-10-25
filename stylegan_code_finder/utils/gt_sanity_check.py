import argparse
from pathlib import Path
from typing import Tuple, Set

import cv2
import numpy
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

ALLOWED_COLORS = {
    (0, 0, 0),
    (255, 0, 0),
    (0, 0, 255),
}


def highlight_errors_in_image(image_array: numpy.ndarray, invalid_colors: Set[Tuple], error_color: Tuple = (0, 255, 0)) \
        -> Image.Image:
    highlighted_image = image_array.copy()
    for color in invalid_colors:
        x, y = numpy.where(numpy.all(image_array == color, axis=-1))
        highlighted_image[x, y] = error_color
    return Image.fromarray(highlighted_image)


def check_colors(image_array: numpy.ndarray, highlight_errors: bool = False, image_path: Path = None,
                 out_dir: Path = None):
    unique_colors = numpy.unique(image_array.reshape(-1, image_array.shape[-1]), axis=0)
    color_palette = set([tuple(c) for c in unique_colors])
    invalid_colors = color_palette - ALLOWED_COLORS
    if len(invalid_colors) > 0:
        print(f"{image_path} contains the following invalid colors: {invalid_colors}")
        if highlight_errors:
            assert image_path is not None and out_dir is not None, "If errors should be highlighted, the required " \
                                                                   "paths have to be set"
            highlighted_image = highlight_errors_in_image(image_array, invalid_colors)
            out_path = out_dir / f"tmp/{image_path.stem}_highlighted{image_path.suffix}"
            highlighted_image.save(str(out_path))


def postprocess_images(image_array: numpy.ndarray, image_path: Path, out_dir: Path, max_contour_area: int = 7):
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(grayscale_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    small_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max_contour_area:
            mask = numpy.zeros(image_array.shape[:2], numpy.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            mean = cv2.mean(image_array, mask=mask)
            if mean[:3] in ALLOWED_COLORS:  # Only take "clean" contours, i.e., contours that contain a single color
                small_contours.append(contour)

    cv2.drawContours(image_array, small_contours, -1, color=(255, 0, 255), thickness=cv2.FILLED)
    out_path = out_dir / f"{image_path.stem}_post{image_path.suffix}"
    cv2.imwrite(str(out_path), image_array)


def main(args: argparse.Namespace):
    if not (args.check_colors or args.postprocess):
        print("You should pass at least one of the following arguments to the script: --check-colors, --postprocess")
        return
    for image_path in tqdm(args.image_dir.iterdir()):
        args.out_dir.mkdir(exist_ok=True)
        try:
            image_array = numpy.array(Image.open(image_path).convert("RGB"))
        except UnidentifiedImageError:
            # print(f"File {image_path} is not an image.")
            continue
        if args.check_colors:
            check_colors(image_array, args.highlight_errors, image_path, args.out_dir)
        if args.postprocess:
            postprocess_images(image_array, image_path, out_dir=args.out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script that checks if the ground truth for segmented documents is "
                                                 "valid and postprocesses it.")
    parser.add_argument("image_dir", type=Path, help="The path where the images that should be checked are stored.")
    parser.add_argument("-cc", "--check-colors", action="store_true", default=False,
                        help="Check images if they contain pixels that are not part of ALLOWED_COLORS")
    parser.add_argument("-he", "--highlight-errors", action="store_true", default=False,
                        help="Highlight pixels that are not part of the list of allowed colors.")
    parser.add_argument("-p", "--postprocess", action="store_true", default=False, help="Show and remove noisy areas")
    parser.add_argument("--out-dir", type=Path, default=Path("tmp"),
                        help="Path to the directory where images should be saved")
    parsed_args = parser.parse_args()
    main(parsed_args)
