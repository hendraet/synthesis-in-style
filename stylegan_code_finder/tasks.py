import io

import msgpack
import os
from io import BytesIO

import celery
from PIL import Image, ImageOps
from celery import Celery


class SegmentationTask(celery.Task):

    def __init__(self):
        pass


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

    image_data = BytesIO(bytes)
    image_data.seek(0)

    with Image.open(image_data) as decoded_image:
        decoded_image = decoded_image.convert('RGB')

        inverted_image = ImageOps.invert(decoded_image)

        byte_array = io.BytesIO()
        inverted_image.save(byte_array, format='PNG')
        byte_array = byte_array.getvalue()

        packed_byte_array = msgpack.packb(byte_array)

    return packed_byte_array
