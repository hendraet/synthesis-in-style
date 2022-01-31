import io
import json
from pathlib import Path

import msgpack
import os
from io import BytesIO

import celery
import torch
from PIL import Image, ImageOps
from celery import Celery

from segmentation.analysis_segmenter import VotingAssemblySegmenter
from segmentation.evaluation.analyze_image_segments import preprocess_images


class SegmentationTask(celery.Task):

    def __init__(self):
        self.config = {
            "model_config_path": os.environ.get("SEGMENTATION_CONFIG_PATH", None),
            "device_id": int(os.environ.get("SEGMENTATION_DEVICE", -1))
        }

        print(self.config)
        assert self.config["model_config_path"] is not None, "You must supply a path to a model configuration in the " \
                                                             "environment variable CLASSIFICATION_CONFIG_PATH "

        if self.config["device_id"] >= 0:
            self.device = torch.device("cuda", self.config["device_id"])
        else:
            self.device = torch.device("cpu")

        self.segmenter = None

    def initialize(self):
        if self.segmenter is not None:
            return

        config_path = Path(self.config["model_config_path"])
        model_config_path = config_path / "config.json"

        with model_config_path.open() as f:
            model_config = json.load(f)

        checkpoint_path = config_path / model_config["checkpoint"]
        color_map_path = config_path / model_config["class_to_color_map"]

        self.segmenter = VotingAssemblySegmenter(
            checkpoint_path,
            "cpu",
            color_map_path,
            max_image_size=int(model_config.get("max_image_size", 0)),
            print_progress=False,
            show_confidence_in_segmentation=False
        )

    @torch.no_grad()
    def predict(self, image):
        assembled_prediction = self.segmenter.segment_image(image)

        return assembled_prediction


broker_address = os.environ.get('BROKER_ADDRESS', 'localhost')
app = Celery('wpi_demo', backend='rpc://', broker=f"pyamqp://guest@{broker_address}//")
app.conf.update(
    accept_content=['msgpack'],
    task_serializer='msgpack',
    result_serializer='msgpack',
)


@app.task(name='text_segmentation', base=SegmentationTask)
def segment(task_data):
    bytes = msgpack.unpackb(task_data)
    segment.initialize()

    image_data = BytesIO(bytes)
    image_data.seek(0)

    with Image.open(image_data) as decoded_image:
        decoded_image = decoded_image.convert('RGB')

        segmentation_results = segment.predict(decoded_image)

        byte_array = io.BytesIO()
        segmentation_results.save(byte_array, format='PNG')
        byte_array = byte_array.getvalue()

        packed_byte_array = msgpack.packb(byte_array)

    return packed_byte_array
