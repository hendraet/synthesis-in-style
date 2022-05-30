import argparse
import functools
import os
from collections import defaultdict
from pathlib import Path
from typing import Union, Dict, Iterable, Type, Callable, List, Optional

import torch
from PIL import Image
from pytorch_training.data.caching_loader import CachingLoader
from pytorch_training.data.json_dataset import JSONDataset
from pytorch_training.data.utils import default_loader
from pytorch_training.distributed import get_world_size, get_rank
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

import global_config
from data.autoencoder_dataset import AutoencoderDataset
from data.base_dataset_gan_dataset import BaseDatasetGANDataset
from data.dataset_gan_dataset import DatasetGANDataset
from data.dataset_gan_generation_dataset import DatasetGANGenerationDataset
from data.segmentation_dataset import AugmentedSegmentationDataset
from networks import load_autoencoder_or_generator
from utils.config import load_config


def resilient_loader(path):
    try:
        return default_loader(path)
    except Exception as e:
        print(f"Could not load {path} with expeption: {e}")
        return Image.new('RGB', (256, 256))


def build_data_loader(image_path: Union[str, Path], config: dict, uses_absolute_paths: bool, shuffle_off: bool = False,
                      dataset_class: Type[JSONDataset] = AutoencoderDataset,
                      loader_func: Callable = resilient_loader, drop_last: bool = True, collate_func: Callable = None) -> DataLoader:
    transform_list = [
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * config['input_dim'], (0.5,) * config['input_dim'])
    ]
    transform_list = transforms.Compose(transform_list)

    dataset = dataset_class(
        image_path,
        root=os.path.dirname(image_path) if not uses_absolute_paths else None,
        transforms=transform_list,
        loader=loader_func,
    )

    sampler = None
    if get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=not shuffle_off)
        sampler.set_epoch(get_rank())

    if shuffle_off:
        shuffle = False
    else:
        shuffle = sampler is None

    if not global_config.debug:
        data_loader_class = functools.partial(DataLoader, num_workers=config['num_workers'])
    else:
        # Specifying 'num_workers' may have a negative impact on debugging capabilities of PyCharm
        data_loader_class = DataLoader

    loader = data_loader_class(
        dataset,
        config['batch_size'],
        shuffle=shuffle,
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=collate_func,
    )
    return loader


def build_dataset_gan_loader(json_path: Union[str, Path], tensor_path: Union[str, Path],
                             class_to_color_map_path: Union[str, Path],
                             config: dict, uses_absolute_paths: bool, shuffle_off: bool = False,
                             dataset_class=Type[BaseDatasetGANDataset],
                             loader_func: Callable = resilient_loader, generator=None,
                             drop_last: bool = True, collate_func: Callable = None) -> DataLoader:
    if dataset_class == DatasetGANGenerationDataset:
        partial_dataset = functools.partial(dataset_class, generator_model=generator)
    elif dataset_class == DatasetGANDataset:
        partial_dataset = DatasetGANDataset
    else:
        raise NotImplementedError
    dataset = partial_dataset(
        json_path,
        tensor_path=tensor_path,
        image_size=config["image_size"],
        upsample_mode=config["upsample_mode"],
        class_probabilities=config["class_probability"],
        root=os.path.dirname(json_path) if not uses_absolute_paths else None,
        loader=loader_func,
        random_sampling=config["random_sampling"],
        class_to_color_map_path=class_to_color_map_path,
    )

    sampler = None
    if get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=not shuffle_off)
        sampler.set_epoch(get_rank())

    if shuffle_off:
        shuffle = False
    else:
        shuffle = sampler is None
    loader = DataLoader(
        dataset,
        config['batch_size'],
        shuffle=shuffle,
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=collate_func,
    )
    return loader


def get_data_loader(dataset_json_path: Path, dataset_name: str, args: argparse.Namespace, config: dict,
                    validation: bool = False, original_generator_config_path: Optional[Path] = None) -> DataLoader:
    if args.cache_root is not None:
        loader_func = CachingLoader(dataset_json_path.parent, args.cache_root, base_loader=resilient_loader)
    else:
        loader_func = resilient_loader

    if dataset_name == 'wpi':
        dataset_class = functools.partial(AugmentedSegmentationDataset,
                                          class_to_color_map_path=Path(args.class_to_color_map),
                                          image_size=config['image_size'],
                                          num_augmentations=config['num_augmentations'])
        data_loader = build_data_loader(dataset_json_path, config, False, shuffle_off=validation,
                                        dataset_class=dataset_class, drop_last=(not validation),
                                        loader_func=loader_func)
    elif config['dataset'] == 'dataset_gan':
        if config["generate"]:
            args.device = "cuda"
            generator_config = load_config(config["checkpoint"], original_generator_config_path)
            args.checkpoint = config["checkpoint"]
            generator = load_autoencoder_or_generator(args, generator_config)
            generator.eval()
            data_loader = build_dataset_gan_loader(dataset_json_path, config["tensor_path"], Path(args.class_to_color_map),
                                                   config, False, shuffle_off=validation,
                                                   dataset_class=DatasetGANGenerationDataset, generator=generator,
                                                   drop_last=(not validation), loader_func=loader_func)
        else:
            data_loader = build_dataset_gan_loader(args.train_json, config["tensor_path"], Path(args.class_to_color_map),
                                                   config, False, shuffle_off=validation, dataset_class=DatasetGANDataset,
                                                   drop_last=(not validation), loader_func=loader_func)
    else:
        raise NotImplementedError

    return data_loader


def fill_plot_images(data_loader: Iterable, num_desired_images: int = 16) -> Dict[str, List[torch.Tensor]]:
    """
        Gathers images to be used with ImagePlotter
    """
    image_list = defaultdict(list)
    for batch in data_loader:
        for image_key, images in batch.items():
            num_images = 0
            for image in images:
                image_list[image_key].append(image)
                num_images += 1
                if num_images >= num_desired_images:
                    break
            if len(image_list.keys()) == len(batch.keys()) and \
                    all([len(v) >= num_desired_images for v in image_list.values()]):
                return image_list
    raise RuntimeError(f"Could not gather enough plot images for display size {num_desired_images}.")
