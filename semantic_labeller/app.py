import copy
import json
import pickle
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageColor
from flask import Flask, send_file, render_template, request

COLOR_MAP = [
    "#00B3FF",  # Vivid Yellow
    "#753E80",  # Strong Purple
    "#0068FF",  # Vivid Orange
    "#D7BDA6",  # Very Light Blue
    "#2000C1",  # Vivid Red
    "#62A2CE",  # Grayish Yellow
    "#667081",  # Medium Gray

    # The following don't work well for people with defective color vision
    "#347D00",  # Vivid Green
    "#8E76F6",  # Strong Purplish Pink
    "#8A5300",  # Strong Blue
    "#5C7AFF",  # Strong Yellowish Pink
    "#7A3753",  # Strong Violet
    "#008EFF",  # Vivid Orange Yellow
    "#5128B3",  # Strong Purplish Red
    "#00C8F4",  # Vivid Greenish Yellow
    "#0D187F",  # Strong Reddish Brown
    "#00AA93",  # Vivid Yellowish Green
    "#153359",  # Deep Yellowish Brown
    "#133AF1",  # Vivid Reddish Orange
    "#162C23",  # Dark Olive Green
]

current_path = Path(__file__)
code_finder_root_path = current_path.parent.parent / '../stylegan_code_finder'
sys.path.append("../stylegan_code_finder")


class Labeller:

    def __init__(self, base_dir: Path, num_clusters: int, class_colors: Path, max_size: int = 256):
        self.base_dir = base_dir
        self.num_clusters = num_clusters
        self.id_size_map = None
        self.arrays = self.init_arrays()
        self.catalogs = self.init_catalogs()
        self.class_colors = self.load_class_colors(class_colors)
        self.color_map = copy.copy(COLOR_MAP)
        self.adjust_color_map()
        self.color_map.extend(self.class_colors.values())
        self.label_map = self.create_label_map()
        self.max_size = max_size

    def create_label_map(self) -> List[dict]:
        # check if we already created a label map for this number of classes and load it instead
        default_save_path = self.base_dir / self.default_result_file_name
        if default_save_path.exists():
            with default_save_path.open() as f:
                label_map = json.load(f)

            # we need to convert all entries with values such as `background` to numbers.
            label_map_with_indices = []
            color_map_length_without_labels = len(self.color_map) - len(self.class_colors)
            for sub_label_map in label_map.values():
                new_sub_label_map = {}
                for key, value in sub_label_map.items():
                    if value in self.class_colors:
                        value = list(self.class_colors.keys()).index(value) + color_map_length_without_labels
                    new_sub_label_map[int(key)] = int(value)
                label_map_with_indices.append(new_sub_label_map)
            return label_map_with_indices
        else:
            return [{i: i for i in range(len(self.color_map))} for _ in range(len(self.catalogs))]

    def adjust_color_map(self):
        if len(self.color_map) < self.num_clusters:
            for i in range(self.num_clusters - len(self.color_map)):
                self.color_map.append(self.color_map[i % len(self.color_map)])

    @staticmethod
    def load_class_colors(class_colors: Path) -> dict:
        with class_colors.open() as f:
            return json.load(f)

    def get_color(self, index: int, image_id: int) -> tuple:
        return ImageColor.getrgb(self.color_map[self.label_map[image_id][index] % len(self.color_map)])

    def get_color_range(self, num_colors: int) -> Dict[int, str]:
        return {len(self.color_map) - len(self.class_colors) + i: color for i, color in
                enumerate(self.class_colors.values())}

    def init_arrays(self) -> List[np.ndarray]:
        file_name = self.base_dir / 'cluster_arrays' / f'{self.num_clusters}.npz'
        npz_data = np.load(str(file_name))
        cluster_images = [npz_data[array] for array in npz_data.files]
        return cluster_images

    def init_catalogs(self) -> list:
        file_name = self.base_dir / 'catalogs' / f'{self.num_clusters}.pkl'
        with file_name.open('rb') as f:
            try:
                catalogs = pickle.load(f)
            except ModuleNotFoundError:
                # we are loading a legacy catalog -> we need to adjust the module paths
                import sys
                from segmentation import gan_local_edit
                sys.modules['gan_local_edit'] = gan_local_edit
                catalogs = pickle.load(f)
            self.id_size_map = catalogs.pop('id_to_size_map')
            catalogs = list(catalogs.values())

        for catalog, array in zip(catalogs, self.arrays):
            num_images, num_channels, height, width = array.shape
            catalog.labels = np.copy(catalog._factorization.labels_.reshape((num_images, height, width)), order='C')

        return catalogs

    @property
    def num_images(self):
        return self.arrays[0].shape[0]

    @property
    def image_size(self):
        largest_array = max(self.arrays, key=lambda x: x.shape[-1])
        largest_array_size = largest_array.shape[-1]
        return min(largest_array_size, self.max_size)

    @property
    def default_result_file_name(self):
        return f'merged_classes_{self.num_clusters}.json'

    def save(self, data: dict):
        file_name = Path(data.get('file_name', self.default_result_file_name))

        adjusted_label_map = []
        for sub_label_map in self.label_map:
            labels = {}
            for key, value in sub_label_map.items():
                color_map_length_without_labels = len(self.color_map) - len(self.class_colors)
                if value >= color_map_length_without_labels:
                    value = list(self.class_colors.keys())[value - color_map_length_without_labels]
                labels[key] = value
            adjusted_label_map.append(labels)

        label_map_to_save = {i: labels for i, labels in zip(self.id_size_map.keys(), adjusted_label_map)}

        with (self.base_dir / file_name.name).open('w') as f:
            json.dump(label_map_to_save, f, indent='\t')

    def label_image_to_rgb(self, image_data: np.ndarray, image_id: int) -> np.ndarray:
        height, width = image_data.shape[-2:]
        result_image = np.empty((3, height, width), dtype=np.uint8)
        for i in range(self.num_clusters):
            color = self.get_color(i, image_id)
            color_image = np.empty_like(result_image)
            for j in range(3):
                color_image[j, ...] = color[j]

            mask_image = image_data == i
            result_image = np.where(mask_image, color_image, result_image)

        return result_image

    def get_image(self, image_id: int, sub_image_id: int, original: bool = False) -> Image:
        if original:
            image_data = self.arrays[sub_image_id][image_id]
        else:
            try:
                image_data = self.catalogs[sub_image_id].labels[image_id % self.num_images]
                image_data = self.label_image_to_rgb(image_data, sub_image_id)
            except IndexError:
                image_data = self.arrays[sub_image_id][image_id]

        image = Image.fromarray(image_data.astype('uint8').transpose(1, 2, 0))
        image = image.resize(((size := self.image_size), size), Image.NEAREST)
        return image

    def adjust_label(self, image_id: int, sub_image_id: int, adjustment_data: dict):
        original_catalog_data = self.catalogs[sub_image_id].labels[image_id % self.num_images]
        catalog_height, catalog_width = original_catalog_data.shape[-2:]
        sample_x_position = int(adjustment_data['position']['x'] / self.image_size * catalog_width)
        sample_y_position = int(adjustment_data['position']['y'] / self.image_size * catalog_height)

        original_label = int(original_catalog_data[sample_y_position, sample_x_position])

        if adjustment_data['mode'] == 'reset':
            self.label_map[sub_image_id][original_label] = int(original_label)
        else:
            self.label_map[sub_image_id][original_label] = int(adjustment_data['color'])


with open('configs/server_config.json') as f:
    server_config = json.load(f)

app = Flask(__name__)
labeller = Labeller(
    Path(server_config['base_dir']),
    int(server_config['num_clusters']),
    Path(server_config.get('class_colors', ''))
)


@app.route('/')
def main():
    return render_template(
        'base.html',
        start_image=0,
        sub_images_per_image=len(labeller.arrays),
        labeller=labeller,
        colors=labeller.get_color_range(labeller.num_clusters)
    )


@app.route('/image/<int:image_id>/<int:sub_image_id>')
def image(image_id: int, sub_image_id: int):
    img_io = BytesIO()
    labeller.get_image(image_id, sub_image_id).save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png'), {'Cache-control': 'no-cache'}


@app.route('/original-image/<int:image_id>/<int:sub_image_id>')
def original_image(image_id: int, sub_image_id: int):
    img_io = BytesIO()
    labeller.get_image(image_id, sub_image_id, original=True).save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png'), {'Cache-control': 'no-cache'}


@app.route('/label/<int:image_id>/<int:sub_image_id>', methods=['POST'])
def label(image_id: int, sub_image_id: int):
    data = json.loads(request.data.decode(request.charset))
    labeller.adjust_label(image_id, sub_image_id, data)
    return {'sub_image_id': sub_image_id}


@app.route('/save', methods=['POST'])
def save():
    data = json.loads(request.data.decode(request.charset))
    labeller.save(data)
    return ''
