import argparse
import datetime
import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from pytorch_training.distributed import synchronize
from pytorch_training.extensions.logger import WandBLogger
from pytorch_training.extensions.lr_scheduler import LRScheduler
from pytorch_training.trainer import DistributedTrainer, Trainer
from pytorch_training.triggers import get_trigger
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import global_config
from training_builder.base_train_builder import BaseTrainBuilder
from training_builder.train_builder_selection import get_train_builder_class
from utils.clamped_cosine import ClampedCosineAnnealingLR
from utils.config import load_yaml_config, merge_config_and_args
from utils.data_loading import get_data_loader


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


def get_scheduler(config: dict, trainer: Trainer, training_builder: BaseTrainBuilder) -> LRScheduler:
    if 'cosine_max_update_epoch' in config:
        cosine_end_iteration = config['cosine_max_update_epoch'] * trainer.iterations_per_epoch
    elif 'cosine_max_update_iter' in config:
        cosine_end_iteration = config['cosine_max_update_iter']
    else:
        cosine_end_iteration = config['epochs']

    schedulers = {}
    for optimizer_name, optimizer in training_builder.get_optimizers().items():
        if config["warm_restarts"]:
            schedulers[optimizer_name] = CosineAnnealingWarmRestarts(optimizer, cosine_end_iteration,
                                                                     eta_min=config['end_lr'])
        else:
            schedulers[optimizer_name] = ClampedCosineAnnealingLR(optimizer, cosine_end_iteration,
                                                                  eta_min=config['end_lr'])

    return LRScheduler(schedulers, trigger=get_trigger((1, 'iteration')))


def setup_distributed(mpi_backend: str, rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(mpi_backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank: int, args: argparse.Namespace, world_size: int):
    if world_size == 1 and os.environ.get('REMOTE_PYCHARM_DEBUG_SESSION', False):
        import pydevd_pycharm

        logging.info('Connecting to debugger...')
        pydevd_pycharm.settrace('localhost', port=int(os.environ.get('REMOTE_PYCHARM_DEBUG_PORT')),
                                stdoutToServer=True, stderrToServer=True, suspend=False)
        logging.info('Done')
    global_config.debug = args.debug

    config = load_yaml_config(args.config)
    config = merge_config_and_args(config, args)
    sanity_check_config(config)

    if world_size > 1:
        setup_distributed(args.mpi_backend, rank, world_size)

    train_data_loader = get_data_loader(Path(config['train_json']), config['dataset'], args, config,
                                        original_generator_config_path=args.original_generator_config_path)
    if args.validation_json is not None:
        val_data_loader = get_data_loader(Path(config['validation_json']), config['dataset'], args, config,
                                          validation=True,
                                          original_generator_config_path=args.original_generator_config_path)
    else:
        val_data_loader = None

    train_builder_class = get_train_builder_class(config)
    training_builder = train_builder_class(config, train_data_loader, val_data_loader, rank=rank, world_size=world_size)

    if 'max_iter' in config:
        stop_trigger = (config['max_iter'], 'iteration')
    else:
        stop_trigger = (config['epochs'], 'epoch')

    trainer = DistributedTrainer(
        training_builder.get_updater(),
        stop_trigger=get_trigger(stop_trigger)
    )

    logging.info("Initializing wandb... ")
    logger = WandBLogger(
        args.log_dir,
        args,
        config,
        os.path.dirname(os.path.realpath(__file__)),
        trigger=get_trigger((config['log_iter'], 'iteration')),
        master=rank == 0,
        project_name=args.wandb_project_name,
        entity=args.wandb_entity,
        run_name=args.log_name,
        disabled=global_config.debug
    )
    logging.info("done")
    evaluator = training_builder.get_evaluator(logger)
    if evaluator is not None:
        trainer.extend(evaluator)

    snapshotter = training_builder.get_snapshotter()
    if snapshotter is not None:
        trainer.extend(snapshotter)

    image_plotter = training_builder.get_image_plotter()
    if image_plotter is not None:
        trainer.extend(image_plotter)

    lr_scheduler = get_scheduler(config, trainer, training_builder)
    trainer.extend(lr_scheduler)

    trainer.extend(logger)

    synchronize()
    logging.info('Setup complete. Starting training...')
    try:
        trainer.train()
    finally:
        wandb.finish()
        if world_size > 1:
            cleanup()
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
    parser.add_argument('--coco-gt', help='PAth to COCO GT required, if you set validation images')
    parser.add_argument('--fine-tune', help='Path to model to finetune from')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mpi-backend', default='nccl', choices=['nccl', 'gloo'],
                        help='MPI backend to use for interprocess communication')
    parser.add_argument('--class-to-color-map', default='handwriting_colors.json',
                        help='path to json file with class color map')
    parser.add_argument('-c', '--cache-root',
                        help='path to a folder where you want to cache images on the local file system')
    parser.add_argument('-l', '--log-dir', default='training', help='outputs path')
    parser.add_argument('-ln', '--log-name', default='training', help='name of the train run')
    parser.add_argument('--warm-restarts', action='store_true', default=False,
                        help='If the scheduler should use warm restarts')
    parser.add_argument('--wandb-project-name', default='Debug', help='The project name of the WandB project')
    parser.add_argument('--wandb-entity', help='The name of the WandB entity')
    parser.add_argument('--debug', action='store_true', default=False, help='Special mode for faster debugging')

    parsed_args = parser.parse_args()
    parsed_args.log_dir = os.path.join('logs', parsed_args.log_dir, parsed_args.log_name,
                                       datetime.datetime.now().isoformat())

    world_size = torch.cuda.device_count()
    logging.info(f"Running on {world_size} GPU(s)")
    torch.multiprocessing.spawn(main, args=(parsed_args, world_size), nprocs=world_size, join=True)
