from typing import List, Optional, Tuple

import cv2
import numpy
from scipy.spatial import ConvexHull

from stylegan_code_finder.utils.segmentation_utils import BBox


class LineBBox:
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

    def as_points(self) -> Tuple:
        return self.bbox.as_points()

    def __iter__(self):
        return iter(self.bbox)

    def __str__(self):
        return f"LineBBox(({self.left}, {self.top}, {self.right}, {self.bottom}), {self.line_candidates})"

    def __repr__(self):
        return f"LineBBox({self.left}, {self.top}, {self.right}, {self.bottom}, {self.line_candidates})"

    def to_dict(self):
        return {
            'bbox': tuple(self.bbox),
            'is_ambiguous': self.is_ambiguous(),
            'line_candidates': self.line_candidates
        }


def get_mutual_bbox(bboxes):
    l, t, r, b = zip(*bboxes)
    return LineBBox(min(l), min(t), max(r), max(b))


def get_min_bbox(line_infos) -> Tuple[numpy.ndarray, Tuple[float, float], float]:
    line_points = []
    bbox_points = [LineBBox(*bbox).as_points() for bbox in line_infos['bboxes']]
    for bbox in bbox_points:
        line_points.extend(bbox)
    line_points = numpy.asarray(line_points)
    convex_hull = ConvexHull(line_points)
    hull_points = line_points[convex_hull.vertices]
    min_area_rect = cv2.minAreaRect(hull_points)
    mid_point, rotation = min_area_rect[1:]
    min_bbox = numpy.int0(cv2.boxPoints(min_area_rect))
    return min_bbox, mid_point, rotation


def radians_to_degrees(rad: float) -> float:
    return rad * 180 / numpy.pi


def degrees_to_radians(degree: float) -> float:
    return degree * numpy.pi / 180


def get_line_rotation(line_coords: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
    x_len = line_coords[1][0] - line_coords[0][0]
    y_len = line_coords[1][1] - line_coords[0][1]
    rad_angle = -numpy.arctan2(y_len, x_len)
    return rad_angle


def rotate_polygon(points: numpy.ndarray, anchor_point: numpy.ndarray, angle: float) -> List[Tuple[int, int]]:
    # Adapted from: https://gis.stackexchange.com/a/23627
    rotated_polygons = numpy.dot(
        points - anchor_point,
        numpy.array([[numpy.cos(angle), numpy.sin(angle)], [-numpy.sin(angle), numpy.cos(angle)]])
    ) + anchor_point
    return [tuple(point) for point in rotated_polygons.astype(numpy.int32)]
