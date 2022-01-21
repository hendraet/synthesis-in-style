from collections import defaultdict
from typing import Tuple, Iterator, List, Dict, Union

import cv2
import numpy
import pytest
from PIL import Image, ImageDraw

from segmentation.base_dataset_segmenter import BaseDatasetSegmenter
from utils.segmentation_utils import draw_contours_on_same_sized_canvases


class MergeTestDatasetSegmenter(BaseDatasetSegmenter):
    def __init__(self):
        # The functionality we want to test is independent of Segmenter state so we can override the initialization
        # function to avoid unnecessary overhead.
        pass


def contour_from_polygon(polygon: List[Tuple[int, int]]) -> numpy.ndarray:
    x_max, y_max = numpy.asarray(polygon).max(axis=0)
    img = Image.new("L", (x_max + 1, y_max + 1))
    draw = ImageDraw.Draw(img)
    draw.polygon(polygon, fill=255)
    contour = cv2.findContours(numpy.asarray(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    return contour


class BaseTestContourMerging:
    def _format_input(self, input_polygons: List[List[List[Tuple[int, int]]]]) \
                      -> Dict[str, Dict[str, List[List[numpy.ndarray]]]]:
        contour_dict = {}
        for i, polygons_for_sub_image in enumerate(input_polygons):
            contour_dict[str(i)] = {
                "printed_text": [
                    [contour_from_polygon(polygon) for polygon in polygons_for_sub_image]
                ]
            }
        return contour_dict

    def _format_result(self, result: List[List[List[Tuple[int, int]]]]) \
                       -> Dict[bool, List[Union[List[numpy.ndarray], None]]]:
        if result[0] is None and result[1] is None:
            return {False: [None], True: [None]}
        resulting_contours = {
            False: [[contour_from_polygon(polygon) for polygon in result[0] + result[1]]],
            True: [[contour_from_polygon(polygon) for polygon in result[0]]]
        }
        return resulting_contours

    def _results_equal(self, contours_a: List[numpy.ndarray], contours_b: List[numpy.ndarray]) -> bool:
        if contours_a is None and contours_b is None:
            return True

        # Break early if the lengths of the contours arrays don't match
        lens_a = [len(c) for c in contours_a]
        lens_b = [len(c) for c in contours_b]
        if sorted(lens_a) != sorted(lens_b):
            return False

        # Reducing the numpy arrays that contain contours to the sum of x and y points should be a suitable heuristic
        # for sorting them although it might not be perfect in every case
        contours_a.sort(key=lambda x: tuple(x.sum(axis=0)[0]))
        contours_b.sort(key=lambda x: tuple(x.sum(axis=0)[0]))

        for i in range(len(contours_a)):
            images = draw_contours_on_same_sized_canvases([contours_a[i], contours_b[i]])
            if not numpy.array_equal(*images):
                return False
        return True

    @pytest.fixture
    def segmenter(self):
        return MergeTestDatasetSegmenter()


class TestOverlapDetection(BaseTestContourMerging):
    @pytest.fixture(params=[
        ([(5, 5), (15, 5), (5, 15)], [(5, 20), (20, 5), (20, 20)]),
    ])
    def non_overlapping_contours(self, request) -> Iterator[numpy.ndarray]:
        return (contour_from_polygon(polygon) for polygon in request.param)

    @pytest.fixture(params=[
        ([(40, 40), (5, 40), (40, 5)], [(30, 30), (25, 30), (30, 25)]),
        ([(20, 20), (5, 20), (20, 5)], [(15, 15), (15, 35), (35, 15)]),
    ])
    def overlapping_contours(self, request) -> Iterator[numpy.ndarray]:
        return (contour_from_polygon(polygon) for polygon in request.param)

    def test_overlap_determination_no_overlap(self, segmenter, non_overlapping_contours):
        assert segmenter.merge_two_contours_if_overlapping(*non_overlapping_contours) is None

    def test_overlap_determination_overlap(self, segmenter, overlapping_contours):
        resulting_contours = segmenter.merge_two_contours_if_overlapping(*overlapping_contours)
        assert resulting_contours is not None
        assert len(resulting_contours) == 1


class TestMergeContours(BaseTestContourMerging):
    INPUT_CONTOURS_TWO_SUB_IMAGES = [
        # contours of first sub-image
        [
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            [(20, 20), (30, 20), (30, 25), (20, 25)],
            [(105, 105), (115, 105), (105, 115)],
            [(200, 200), (210, 200), (210, 210), (200, 210)],
        ],
        # contours of second sub-image
        [
            [(5, 5), (15, 5), (15, 15), (5, 15)],
            [(0, 40), (40, 40), (40, 50), (0, 50)],
            [(105, 120), (120, 105), (120, 120)],
            [(203, 203), (208, 203), (208, 208), (203, 208)],
        ]
    ]

    RESULTING_CONTOURS_TWO_SUB_IMAGES = [
        # merged overlapping contours
        [
            [(0, 0), (10, 0), (10, 5), (15, 5), (15, 15), (5, 15), (5, 10), (0, 10)],
            [(200, 200), (210, 200), (210, 210), (200, 210)],
        ],
        # non-overlapping contours
        [
            [(20, 20), (30, 20), (30, 25), (20, 25)],
            [(0, 40), (40, 40), (40, 50), (0, 50)],
            [(105, 105), (115, 105), (105, 115)],
            [(105, 120), (120, 105), (120, 120)]
        ]
    ]

    INPUT_CONTOURS_THREE_SUB_IMAGES = [
        [
            [(130, 145), (140, 145), (140, 160), (130, 160)],
            [(200, 200), (210, 200), (210, 220), (200, 220)],
            [(300, 310), (315, 310), (315, 315), (300, 315)],
            [(500, 500), (505, 500), (505, 505), (500, 505)],
            [(750, 740), (770, 740), (770, 760), (750, 760)]
        ],
        [
            [(100, 100), (120, 100), (120, 140), (100, 140)],
            [(138, 110), (150, 110), (150, 163), (138, 163)],
            [(205, 207), (215, 207), (215, 221), (205, 221)],
            [(410, 444), (418, 444), (418, 477), (410, 477)],
            [(500, 500), (505, 500), (505, 505), (500, 505)],
            [(600, 600), (605, 600), (605, 605), (600, 605)]
        ],
        [
            [(115, 130), (135, 130), (135, 150), (115, 150)],
            [(306, 312), (317, 312), (317, 318), (306, 318)],
            [(404, 420), (414, 420), (414, 469), (404, 469)],
            [(808, 888), (888, 888), (888, 898), (808, 898)]
        ]
    ]

    RESULTING_CONTOURS_THREE_SUB_IMAGES = [
        [
            [(100, 100), (120, 100), (120, 130), (135, 130), (135, 145), (138, 145), (138, 110), (150, 110), (150, 163),
             (138, 163), (138, 160), (130, 160), (130, 150), (115, 150), (115, 140), (100, 140)],
            [(200, 200), (210, 200), (210, 207), (215, 207), (215, 221), (205, 221), (205, 220), (200, 220)],
            [(300, 310), (315, 310), (315, 312), (317, 312), (317, 318), (306, 318), (306, 315), (300, 315)],
            [(404, 420), (414, 420), (414, 444), (418, 444), (418, 477), (410, 477), (410, 469), (404, 469)],
            [(500, 500), (505, 500), (505, 505), (500, 505)]
        ],
        [
            [(600, 600), (605, 600), (605, 605), (600, 605)],
            [(750, 740), (770, 740), (770, 760), (750, 760)],
            [(808, 888), (888, 888), (888, 898), (808, 898)]
        ]
    ]

    INPUT_CONTOURS_ONE_SUB_IMAGE_EMPTY = [
        [],
        [
            [(5, 5), (15, 5), (15, 15), (5, 15)],
            [(0, 40), (40, 40), (40, 50), (0, 50)],
        ]
    ]

    RESULTING_CONTOURS_ONE_SUB_IMAGE_EMPTY = [
        [
            [(5, 5), (15, 5), (15, 15), (5, 15)],
            [(0, 40), (40, 40), (40, 50), (0, 50)],
        ],
        []
    ]

    INPUT_BOXES_ALL_SUB_IMAGES_EMPTY = [[], []]
    RESULTING_BOXES_ALL_SUB_IMAGES_EMPTY = [None, None]

    INPUT_CONTOURS_NO_OVERLAP = [
        [
            [(100, 100), (120, 100), (120, 140), (100, 140)],
        ],
        [
            [(404, 420), (414, 420), (414, 469), (404, 469)],
        ],
        [
            [(808, 888), (888, 888), (888, 898), (808, 898)]
        ]
    ]

    @pytest.fixture(params=[
        (INPUT_CONTOURS_TWO_SUB_IMAGES, RESULTING_CONTOURS_TWO_SUB_IMAGES),
        (INPUT_CONTOURS_THREE_SUB_IMAGES, RESULTING_CONTOURS_THREE_SUB_IMAGES),
        (INPUT_CONTOURS_ONE_SUB_IMAGE_EMPTY, RESULTING_CONTOURS_ONE_SUB_IMAGE_EMPTY),
        (INPUT_BOXES_ALL_SUB_IMAGES_EMPTY, RESULTING_BOXES_ALL_SUB_IMAGES_EMPTY),
    ], ids=["two sub-images", "three sub-images", "one sub-image none", "all sub-images none"])
    def contours(self, request):
        input_contours = self._format_input(request.param[0])
        resulting_contours = self._format_result(request.param[1])
        return input_contours, resulting_contours

    @pytest.mark.parametrize("only_keep_overlapping", [True, False], ids=["only keep overlapping", "keep all"])
    def test_merging_multiple_sub_images(self, segmenter, contours, only_keep_overlapping):
        input_contours, resulting_contours = contours
        result = resulting_contours[only_keep_overlapping][0]
        merged_contours = segmenter.merge_contours_of_same_class_from_different_images(input_contours, 1,
                                                                                       only_keep_overlapping,
                                                                                       ("printed_text",))["printed_text"][0]
        if result is None:
            assert merged_contours is None
        else:
            assert len(merged_contours) == len(result)
            assert self._results_equal(result, merged_contours)

    @pytest.mark.parametrize("num_sub_images", [2, 3], ids=["two sub-images", "three sub-images"])
    def test_no_overlap(self, num_sub_images, segmenter):
        input_contours = self._format_input(self.INPUT_CONTOURS_NO_OVERLAP[:num_sub_images])
        merged_contours = segmenter.merge_contours_of_same_class_from_different_images(input_contours, 1, True,
                                                                                       ("printed_text",))["printed_text"][0]
        assert merged_contours is None

    @pytest.fixture(params=[
        (INPUT_CONTOURS_TWO_SUB_IMAGES, RESULTING_CONTOURS_TWO_SUB_IMAGES),
        (INPUT_CONTOURS_THREE_SUB_IMAGES, RESULTING_CONTOURS_THREE_SUB_IMAGES),
    ], ids=["two sub-images", "three sub-images"])
    def contours_with_multiple_batches(self, request):
        input_contours = self._format_input(request.param[0])
        for v in input_contours.values():
            v["printed_text"].append([c + 10 for c in v["printed_text"][0]])
        resulting_contours = self._format_result(request.param[1])
        for k, v in resulting_contours.items():
            resulting_contours[k].append([c + 10 for c in v[0]])
        return input_contours, resulting_contours

    @pytest.mark.parametrize("only_keep_overlapping", [True, False], ids=["only keep overlapping", "keep all"])
    def test_merging_with_multiple_batches(self, only_keep_overlapping, contours_with_multiple_batches, segmenter):
        input_contours, resulting_contours = contours_with_multiple_batches
        result = resulting_contours[only_keep_overlapping]
        batch_size = len(result)
        merged_contours = segmenter.merge_contours_of_same_class_from_different_images(input_contours, batch_size,
                                                                                       only_keep_overlapping,
                                                                                       ("printed_text",))["printed_text"]
        for i in range(batch_size):
            assert self._results_equal(result[i], merged_contours[i])

    @pytest.fixture(params=[
        (INPUT_CONTOURS_TWO_SUB_IMAGES, RESULTING_CONTOURS_TWO_SUB_IMAGES),
        (INPUT_CONTOURS_THREE_SUB_IMAGES, RESULTING_CONTOURS_THREE_SUB_IMAGES),
    ])
    def single_sub_image_contours(self, request):
        input_contours = self._format_input(request.param[0])
        merged_input_contours = defaultdict(list)
        merged_input_contours["printed_text"].append([])
        for v in input_contours.values():
            merged_input_contours["printed_text"][0].extend(v["printed_text"][0])
        resulting_contours = self._format_result(request.param[1])
        return merged_input_contours, resulting_contours

    def test_merging_single_sub_image(self, single_sub_image_contours, segmenter):
        input_boxes, resulting_boxes = single_sub_image_contours
        merged_boxes = segmenter.merge_contours_of_same_class_from_same_image(input_boxes)
        assert self._results_equal(merged_boxes["printed_text"][0], resulting_boxes[False][0])
