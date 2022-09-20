from typing import List, Optional

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

    def to_dict(self):
        return {
            'bbox': tuple(self.bbox),
            'is_ambiguous': self.is_ambiguous(),
            'line_candidates': self.line_candidates
        }
