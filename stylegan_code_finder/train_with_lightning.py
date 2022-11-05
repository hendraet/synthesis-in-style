import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import wandb

import global_config
import pathlib
from lightning_modules.ligntning_module_selection import get_segmenter_class
from utils.config import load_yaml_config, merge_config_and_args
from utils.data_loading import get_data_loader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


def sanity_check_config(config: dict):
    if 'network' in config:
        choices = ['DocUFCN', 'TransUNet', 'EMANet', 'PixelEnsemble']
        assert config['network'] in choices, f'The network must be one of: {", ".join(choices)}'
    if 'dataset' in config:
        choices = ['wpi', 'dataset_gan']
        assert config['dataset'] in choices, f'The dataset must be one of: {", ".join(choices)}'
    with open(config['class_to_color_map']) as f:
        class_to_color_map = json.load(f)
    assert len(class_to_color_map) == config['num_classes'], f'The number of classes in the class_to_color_map must ' \
                                                             f'be equal to the num_classes in the config'


def main(args: argparse.Namespace):
    if os.environ.get('REMOTE_PYCHARM_DEBUG_SESSION', False):
        import pydevd_pycharm

        logging.info('Connecting to debugger...')
        pydevd_pycharm.settrace('localhost', port=int(os.environ.get('REMOTE_PYCHARM_DEBUG_PORT')),
                                stdoutToServer=True, stderrToServer=True, suspend=False)
        logging.info('Done')
    global_config.debug = args.debug

    config = load_yaml_config(args.config)
    config = merge_config_and_args(config, args)
    sanity_check_config(config)

    train_data_loader = get_data_loader(Path(config['train_json']), config['dataset'], args, config,
                                        original_generator_config_path=args.original_generator_config_path)
    if args.validation_json is not None:
        val_data_loader = get_data_loader(Path(config['validation_json']), config['dataset'], args, config,
                                          validation=True,
                                          original_generator_config_path=args.original_generator_config_path)
    else:
        val_data_loader = None

    config['num_iter_epoch'] = len(train_data_loader)
    pathlib.Path(args.log_dir, args.log_name).mkdir(parents=True, exist_ok=True)

    logging.info("Initializing wandb... ")
    logger = WandbLogger(name=args.log_name, save_dir=args.log_dir, project=args.wandb_project_name)
    logging.info("done")

    segmenter_class = get_segmenter_class(config)
    segmenter = segmenter_class(config)

    logging.info('Setup complete. Starting training...')
    try:
        if 'max_iter' in config:
            segmentation_trainer = pl.Trainer(strategy="ddp_find_unused_parameters_false", logger=logger,
                                              max_steps=config['max_iter'], accelerator='gpu', devices='auto')
        else:
            segmentation_trainer = pl.Trainer(strategy="ddp_find_unused_parameters_false", logger=logger,
                                              max_epochs=config['epochs'], accelerator='gpu', devices='auto')
        segmentation_trainer.fit(segmenter, train_data_loader, val_data_loader)
    finally:
        wandb.finish()
    logging.info('Training finished')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('Training script started')
    parser = argparse.ArgumentParser(description='Train a network for semantic segmentation of documents')
    parser.add_argument('config', help='path to config with common train settings, such as LR')
    parser.add_argument("-op", "--original-generator-config-path", type=Path, default=None,
                        help="Path to the YAML/JSON file that contains the config for the original segmenter "
                             "training Has to be provided if model was not trained and the original logging "
                             "structure is not present, i.e. the config does not lie in a sibling directory of the "
                             "checkpoint.")
    parser.add_argument('--images', dest='train_json', required=True,
                        help='Path to json file with train images')
    parser.add_argument('--val-images', dest='validation_json',
                        help='path to json file with validation images')
    parser.add_argument('--fine-tune', help='Path to model to finetune from')
    parser.add_argument('--class-to-color-map', default='handwriting_colors.json',
                        help='path to json file with class color map')
    parser.add_argument('-c', '--cache-root',
                        help='path to a folder where you want to cache images on the local file system')
    parser.add_argument('-l', '--log-dir', default='training', help='outputs path')
    parser.add_argument('-ln', '--log-name', default='training', help='name of the train run')
    parser.add_argument('--warm-restarts', action='store_true', default=False,
                        help='If the scheduler should use warm restarts')
    parser.add_argument('--wandb-project-name', default='Debug', help='The project name of the WandB project')
    parser.add_argument('--debug', action='store_true', default=False, help='Special mode for faster debugging')
    parser.add_argument('--num_val', type=int, dest='num_val_visualization', default=2, help='number of validation batch images to send to wandb')

    parsed_args = parser.parse_args()
    parsed_args.log_dir = Path('logs', parsed_args.log_dir, parsed_args.log_name, datetime.datetime.now().isoformat())
    main(args=parsed_args)
