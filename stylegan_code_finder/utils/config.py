import argparse
import json
from pathlib import Path
from typing import Union

import yaml


def load_config_from_alternative_file(config_path, checkpoint_path, insert_stylegan_checkpoint: bool = False):
    with Path(config_path).open() as f:
        if config_path.suffix == '.json':
            config = json.load(f)
        elif config_path.suffix == '.yaml':
            config = yaml.safe_load(f)
        else:
            raise NotImplementedError

    if insert_stylegan_checkpoint:
        if checkpoint_path is not None:
            config['stylegan_checkpoint'] = checkpoint_path
        assert config.get('stylegan_checkpoint', None) is not None

    return config


def load_config_from_checkpoint(checkpoint_path):
    config_dir = Path(checkpoint_path).parent.parent / 'config'
    original_config = config_dir / 'config.json'

    try:
        with open(original_config) as f:
            original_config = json.load(f)

        original_args = config_dir / 'args.json'
        with open(original_args) as f:
            original_args = json.load(f)

        original_config.update(original_args)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            'When trying to load a model form a checkpoint assert that the original configs are in ../config. '
            'Otherwise use the corresponding flag to pass the original config directly.'
        ) from err

    return original_config


def load_config(checkpoint_path: str = None, config_path: Union[str, Path] = None,
                insert_stylegan_checkpoint: bool = False) -> dict:
    if checkpoint_path is None and config_path is None:
        raise RuntimeError("You have to supply either checkpoint path or path to a config file!")

    if config_path is not None:
        config = load_config_from_alternative_file(config_path, checkpoint_path, insert_stylegan_checkpoint)
    else:
        config = load_config_from_checkpoint(checkpoint_path)

    return config


def load_yaml_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_config_and_args(config: dict, args: argparse.Namespace) -> dict:
    for key in dir(args):
        if key.startswith("_"):
            continue
        config[key] = getattr(args, key)
    return config
