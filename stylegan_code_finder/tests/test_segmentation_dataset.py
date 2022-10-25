import numpy
import pytest
import torch

from data.segmentation_dataset import SegmentationDataset


class DatasetFake(SegmentationDataset):
    def __init__(self):
        self.class_to_color_map = {
            "printed_text": "#0000FF",
            "handwritten_text": "#FF0000",
            "background": "#000000",
        }
        self.background_class_name = "background"
        self.image_size = None


class TestSegmentationDataset:
    @pytest.fixture()
    def segmentation_image(self):
        image = [
            [[0, 0, 0],   [255, 0, 0], [0, 0, 0],   [0, 0, 0]  ],
            [[0, 0, 0],   [255, 0, 0], [0, 0, 0],   [0, 0, 0]  ],
            [[0, 0, 255], [0, 0, 0],   [0, 0, 0],   [0, 0, 255]],
            [[0, 0, 0],   [255, 0, 0], [255, 0, 0], [0, 0, 0]  ],
        ]
        return numpy.array(image)

    @pytest.fixture()
    def expected_segmentation_tensor(self):
        tensor = [[
            [0, 2, 0, 0],
            [0, 2, 0, 0],
            [1, 0, 0, 1],
            [0, 2, 2, 0],
        ]]
        return torch.tensor(tensor).type(torch.LongTensor)

    @pytest.fixture
    def dataset(self):
        return DatasetFake()

    def test_segmentation_image_parsing(self, dataset, segmentation_image, expected_segmentation_tensor):
        segmentation_tensor = dataset.segmentation_image_to_class_image(segmentation_image)
        segmentation_tensor = dataset.class_image_to_tensor(segmentation_tensor).type(torch.LongTensor)
        assert torch.equal(segmentation_tensor, expected_segmentation_tensor)


