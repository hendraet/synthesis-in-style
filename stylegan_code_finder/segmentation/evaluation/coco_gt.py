import argparse
import datetime
import json
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import cv2
import numpy
import pycocotools.mask
from PIL import Image, ImageColor
from tqdm.contrib import tenumerate


class COCOGtCreator:

    def __init__(self, class_to_color_map: Dict, image_root: Path = Path('/')):
        self.class_to_color_map = class_to_color_map
        self.categories = self.build_categories()
        self.image_root = image_root

    def build_categories(self) -> List[dict]:
        categories = []
        for category_id, (class_name, color) in enumerate(self.class_to_color_map.items()):
            categories.append({
                "id": category_id,
                "name": class_name,
                "supercategory": class_name,
                "color": color
            })
        return categories

    @staticmethod
    def get_label_image(image_data: Image) -> numpy.ndarray:
        image_data = numpy.array(image_data)
        _, label_image = numpy.split(image_data, 2, axis=1)
        return label_image

    @staticmethod
    def extract_rle(class_mask: numpy.ndarray) -> List[dict]:
        contours, _ = cv2.findContours(class_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rles = []
        for contour in contours:
            if contour.size >= 6:
                rles.append(contour.ravel())

        if len(rles) == 0:
            return rles
        rles = pycocotools.mask.frPyObjects(rles, class_mask.shape[-2], class_mask.shape[-1])
        return rles

    def determine_classes_in_image(self, image_data: Image) -> Dict[str, bool]:
        label_image = self.get_label_image(image_data)
        classes_in_image = {}
        for class_id, (class_name, color) in enumerate(self.class_to_color_map.items()):
            if class_name == 'background':
                continue
            class_mask = numpy.multiply.reduce(label_image[:, :] == ImageColor.getrgb(color), axis=2)
            contours = self.extract_rle(class_mask)
            key = f"has_{class_name}"
            if len(contours) > 0:
                # class is present in image
                classes_in_image[key] = True
            else:
                classes_in_image[key] = False
        return classes_in_image

    def build_annotations_for_image(self, image_data: Image, image_id: int, annotation_id: int) -> Tuple[List[dict], int]:
        label_image = self.get_label_image(image_data)

        annotations = []
        for class_id, (class_name, color) in enumerate(self.class_to_color_map.items()):
            if class_name == 'background':
                # no need to annotate background
                continue
            class_mask = numpy.multiply.reduce(label_image[:, :] == ImageColor.getrgb(color), axis=2)
            rles = self.extract_rle(class_mask)
            if len(rles) > 0:
                for rle in rles:
                    rle_area = int(pycocotools.mask.area(rle))
                    rle['counts'] = rle['counts'].decode('utf-8')
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "segmentation": rle,
                        "area": rle_area,
                        "bbox": pycocotools.mask.toBbox(rle).tolist(),
                        "iscrowd": 0
                    }
                    annotations.append(annotation)
                    annotation_id += 1
        return annotations, annotation_id


    def create_coco_gt_from_image_paths(self, image_paths: Iterable[Path]) -> dict:
        images = []
        annotations = []

        annotation_id = 0
        for i, image_path in tenumerate(image_paths, desc="Create COCO gt"):
            with Image.open(str(image_path)) as the_image:
                image_data = {
                    "id": i,
                    "width": the_image.width // 2,
                    "height": the_image.height,
                    "file_name": str(image_path.relative_to(self.image_root)),
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": str(datetime.datetime.utcnow())
                }
                images.append(image_data)

                annotations_for_image, annotation_id = self.build_annotations_for_image(the_image, i, annotation_id)
                annotations.extend(annotations_for_image)

        coco_gt = {
            "info": {
                "year": datetime.date.today().year,
                "version": "1",
                "description": "COCO GT for evaluation of semantic segmentation",
                "contributor": "yourself",
                "url": "http://example.com",
            },
            "images": images,
            "annotations": annotations,
            "categories": self.categories,
            "licenses": [{
                "id": 0,
                "name": "Kekse",
                "url": "http://example.com"
            }]
        }
        return coco_gt


def iter_through_images_in(image_root: Path, extension: str = 'png') -> Iterable[Path]:
        images = image_root.glob(f'**/*.{extension}')
        for image_path in images:
            yield image_path


def create_coco_gt_from_image_root(image_root: Path, class_to_color_map: Path):

    coco_creator = COCOGtCreator(class_to_color_map)
    image_paths = iter_through_images_in(image_root)
    coco_gt = coco_creator.create_coco_gt_from_image_paths(image_paths)


    with (image_root / 'coco_gt.json').open('w') as f:
        json.dump(coco_gt, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Provide an Image Root with Segmentation images and create COCO GT")
    parser.add_argument("image_root")
    parser.add_argument("class_to_color_map")

    args = parser.parse_args()
    create_coco_gt_from_image_root(Path(args.image_root), Path(args.class_to_color_map))
