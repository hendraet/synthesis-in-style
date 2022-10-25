from collections import defaultdict
from typing import Union, Dict

import torch
from torch.utils.data.dataloader import default_collate

from pytorch_training.data.json_dataset import JSONDataset


class COCODataset(JSONDataset):

    def load_json_data(self, json_data: Union[dict, list]):
        self.images = json_data['images']
        annotation_data = json_data['annotations']
        self.annotations = defaultdict(list)
        for annotation in annotation_data:
            self.annotations[annotation['image_id']].append(annotation)
        self.categories = json_data['categories']

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_func(batch):
        annotations = [d.pop('annotations') for d in batch]
        collated = default_collate(batch)
        collated['annotations'] = annotations
        return collated

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_info = self.images[index]
        path = image_info['file_name']
        if self.root is not None:
            path = self.root / path

        image = self.loader(path)
        image_data = image.crop((0, 0, image.width // 2, image.height))

        if self.transforms is not None:
            image_data = self.transforms(image_data)

        annotations = self.annotations[image_info['id']]
        return {
            "images": image_data,
            "annotations": annotations,
            "image_id": image_info['id']
        }
