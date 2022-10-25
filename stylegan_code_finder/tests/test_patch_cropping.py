import math

import pytest
from typing import Union, Tuple

from segmentation.analysis_segmenter import AnalysisSegmenter


class PatchCropTestAnalysisSegmenter(AnalysisSegmenter):
    def __init__(self, patch_size: int, patch_overlap: Union[int] = 0, patch_overlap_factor: Union[float] = 0.0):
        self.patch_size = patch_size
        self.set_patch_overlap(patch_overlap, patch_overlap_factor)


class TestPatchCropping:
    patch_sizes = [10, 64, 256]
    image_size_factors = [
        (2, 1),
        (2, 2),
        (2.5, 3),
        (3.7, 4.2),
        (2, 8.5),
        (50.3, 100.7),
    ]
    factor_ids = [f"size_factor {int(t[0])}x{int(t[1])}" for t in image_size_factors]

    @pytest.fixture(params=patch_sizes, ids=[f"patch_size {s}" for s in patch_sizes])
    def patch_sizes(self, request):
        return request.param

    @pytest.fixture
    def default_segmenter(self, patch_sizes):
        return PatchCropTestAnalysisSegmenter(patch_sizes)

    @pytest.fixture(params=[0.25, 0.3, 0.5, 0.6, 0.75, 0.9])
    def overlap_factor_segmenter(self, patch_sizes, request):
        return PatchCropTestAnalysisSegmenter(patch_sizes, patch_overlap_factor=request.param)

    @pytest.fixture(params=[1, 3, 5, 6, 7, 9])
    def absolute_overlap_segmenter(self, patch_sizes, request):
        return PatchCropTestAnalysisSegmenter(patch_sizes, patch_overlap=request.param)

    @pytest.mark.parametrize("overlap",
                             [(-1, 0.0), (500, 0.0), (0, -1.0), (0, 1.0), (2, 0.9)],
                             ids=["abs too low", "abs too high", "factor too low", "factor too high", "both specified"])
    def test_wrong_segmenter_instantiation(self, overlap: Tuple[Union[int, None], Union[float, None]]):
        absolute_overlap, overlap_factor = overlap
        with pytest.raises(AssertionError) as exec_info:
            PatchCropTestAnalysisSegmenter(10, patch_overlap=absolute_overlap, patch_overlap_factor=overlap_factor)
        assert exec_info.type == AssertionError

    @pytest.mark.parametrize(
        "image_size_factor", [(1, 1), (2, 1), (1, 2), (3, 3), (1.5, 1), (1.5, 1.5), (4.5, 4.5), (5.7, 6.1)]
    )
    def test_no_specific_overlap(self, default_segmenter: PatchCropTestAnalysisSegmenter,
                                 image_size_factor: Tuple[float, float]):
        image_size = (int(image_size_factor[0] * default_segmenter.patch_size),
                      int(image_size_factor[1] * default_segmenter.patch_size))
        resulting_patches = default_segmenter.calculate_bboxes_for_patches(*image_size)
        assert len(resulting_patches) == math.ceil(image_size_factor[0]) * math.ceil(image_size_factor[1])

    @pytest.mark.parametrize("image_size_factor", image_size_factors, ids=factor_ids)
    def test_overlap_factor_calculation(self, overlap_factor_segmenter: PatchCropTestAnalysisSegmenter,
                                        image_size_factor: Tuple[float, float]):
        # Code duplication between this function and test_absolute_overlap_calculation is intentional since pytest
        # doesn't allow the usage of fixtures in parameterize yet. Otherwise one could do sth. like:
        #   @pytest.mark.parametrize("segmenter", [overlap_factor_segmenter, absolute_overlap_segmenter])
        image_size = (int(image_size_factor[0] * overlap_factor_segmenter.patch_size),
                      int(image_size_factor[1] * overlap_factor_segmenter.patch_size))
        resulting_patches = overlap_factor_segmenter.calculate_bboxes_for_patches(*image_size)

        width_overlap = resulting_patches[0][2] - resulting_patches[1][0]
        assert width_overlap == overlap_factor_segmenter.patch_overlap

        first_box_second_row = [box for box in resulting_patches if box[1] > 0][0]
        height_overlap = resulting_patches[0][3] - first_box_second_row[1]
        assert height_overlap == overlap_factor_segmenter.patch_overlap

    @pytest.mark.parametrize("image_size_factor", image_size_factors, ids=factor_ids)
    def test_absolute_overlap_calculation(self, absolute_overlap_segmenter: PatchCropTestAnalysisSegmenter,
                                          image_size_factor: Tuple[float, float]):
        image_size = (int(image_size_factor[0] * absolute_overlap_segmenter.patch_size),
                      int(image_size_factor[1] * absolute_overlap_segmenter.patch_size))
        resulting_patches = absolute_overlap_segmenter.calculate_bboxes_for_patches(*image_size)

        width_overlap = resulting_patches[0][2] - resulting_patches[1][0]
        assert width_overlap == absolute_overlap_segmenter.patch_overlap

        first_box_second_row = [box for box in resulting_patches if box[1] > 0][0]
        height_overlap = resulting_patches[0][3] - first_box_second_row[1]
        assert height_overlap == absolute_overlap_segmenter.patch_overlap

    @pytest.mark.parametrize("image_size_factor", image_size_factors, ids=factor_ids)
    def test_automatic_patch_calculation(self, default_segmenter: PatchCropTestAnalysisSegmenter,
                                         image_size_factor: Tuple[float, float]):
        image_size = (int(image_size_factor[0] * default_segmenter.patch_size),
                      int(image_size_factor[1] * default_segmenter.patch_size))
        resulting_patches = default_segmenter.calculate_bboxes_for_patches(*image_size)

        if (image_size[0] % default_segmenter.patch_size) == 0:
            assert resulting_patches[-1][2] == image_size[0]
        else:
            assert resulting_patches[-1][2] > image_size[0]

        if (image_size[1] % default_segmenter.patch_size) == 0:
            assert resulting_patches[-1][3] == image_size[1]
        else:
            assert resulting_patches[-1][3] > image_size[1]
