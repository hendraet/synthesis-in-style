import argparse
import datetime
import functools
import multiprocessing
import os
from pathlib import Path

import torch
from pytorch_training.data.caching_loader import CachingLoader
from pytorch_training.data.json_dataset import JSONDataset
from pytorch_training.distributed import synchronize, get_rank, get_world_size
from pytorch_training.extensions import Snapshotter
from pytorch_training.extensions.logger import WandBLogger
from pytorch_training.extensions.lr_scheduler import LRScheduler
from pytorch_training.optimizer import GradientClipAdam
from pytorch_training.trainer import DistributedTrainer
from pytorch_training.triggers import get_trigger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

from extensions.fid_score import FIDScore
from extensions.stylegan_image_plotter import StyleGANImagePlotter
from updater.stylegan_2_updater import Stylegan2Updater as Updater
from utils.config import load_yaml_config, merge_config_and_args
from utils.data_loading import build_data_loader, resilient_loader
from utils.weight_loading import load_weights


def main(args: argparse.Namespace, rank: int, world_size: int):
    config = load_yaml_config(args.config)
    config = merge_config_and_args(config, args)

    if args.cache_root is not None:
        image_loader = CachingLoader(
            os.path.dirname(config['images']),
            cache_root=Path(args.cache_root),
            base_loader=resilient_loader
        )
    else:
        image_loader = resilient_loader

    train_loader = build_data_loader(
        config['images'],
        config,
        False,
        dataset_class=JSONDataset,
        loader_func=image_loader
    )
    val_loader = build_data_loader(config['val_images'], config, False, dataset_class=JSONDataset,
                                   loader_func=image_loader)

    if config['stylegan_variant'] == 'swagan':
        from networks.swagan import Generator, Discriminator
    else:
        from networks.stylegan2 import Generator, Discriminator

    generator = Generator(
        config['image_size'],
        config['latent_size'],
        config['n_mlp'],
        channel_multiplier=config['channel_multiplier']
    )
    discriminator = Discriminator(
        config['image_size'], channel_multiplier=config['channel_multiplier']
    )

    g_ema = Generator(
        config['image_size'],
        config['latent_size'],
        config['n_mlp'],
        channel_multiplier=config['channel_multiplier']
    ).to(config['device'])
    g_ema.eval()

    if config['resume_ckpt'] is not None:
        networks = {'generator': generator, 'discriminator': discriminator, 'g_ema': g_ema}
        load_weights(networks=networks, model_file=config['resume_ckpt'])

    g_reg_ratio = int(config['regularization']['g_interval']) / (int(config['regularization']['g_interval']) + 1)
    d_reg_ratio = int(config['regularization']['d_interval']) / (int(config['regularization']['d_interval']) + 1)

    generator_optimizer = GradientClipAdam(
        generator.parameters(),
        lr=float(config['lr']) * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        # weight_decay=float(config['weight_decay']),
    )
    discriminator_optimizer = GradientClipAdam(
        discriminator.parameters(),
        lr=float(config['lr']) * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        # weight_decay=float(config['weight_decay']),
    )

    if world_size > 1:
        distributed = functools.partial(
            DDP,
            device_ids=[rank],
            # find_unused_parameters=True,
            broadcast_buffers=False,
            output_device=rank
        )
        generator = distributed(generator.to('cuda'))
        if discriminator is not None:
            discriminator = distributed(discriminator.to('cuda'))
    else:
        generator = generator.to('cuda')
        if discriminator is not None:
            discriminator = discriminator.to('cuda')

    updater_kwargs = {
        'iterators': {'images': train_loader},
        'networks': {'generator': generator, 'discriminator': discriminator},
        'optimizers': {'generator': generator_optimizer, 'discriminator': discriminator_optimizer},
        'device': 'cuda',
        'copy_to_device': world_size == 1,
        'regularization_options': config['regularization'],
        'style_mixing_prob': float(config['style_mixing_prob']),
        'latent_size': int(config['latent_size']),
        'g_ema': g_ema,
        'freeze_stochastic_noise_layers': config.get('freeze_stochastic_noise_layers', False)
    }

    updater = Updater(**updater_kwargs)
    updater.accumulate(generator, 0)

    trainer = DistributedTrainer(
        updater,
        stop_trigger=get_trigger((int(config['max_iter']), 'iteration'))
    )

    logger = WandBLogger(
        config['log_dir'],
        args,
        config,
        os.path.dirname(os.path.realpath(__file__)),
        trigger=get_trigger((int(config['log_iter']), 'iteration')),
        master=rank == 0,
        project_name=args.wandb_project_name,
        entity=args.wandb_entity,
        run_name=config['log_name'],
    )

    fid_extension = FIDScore(
        g_ema,
        val_loader,
        dataset_path=config['val_images'],
        trigger=(1, 'epoch'),
    )
    trainer.extend(fid_extension)

    if rank == 0:
        strip_ddp = lambda model: model if not isinstance(model, DDP) else model.module
        snapshotter = Snapshotter(
            {
                'generator': strip_ddp(generator),
                'discriminator': strip_ddp(discriminator),
                'g_ema': g_ema,
                'generator_optimizer': generator_optimizer,
                'discriminator_optimizer': discriminator_optimizer
            },
            args.log_dir,
            trigger=get_trigger((config['snapshot_save_iter'], 'iteration'))
        )
        trainer.extend(snapshotter)

        sample_z = torch.randn(int(config['batch_size']), int(config['latent_size']), device=config['device']).unbind(0)
        plotter = StyleGANImagePlotter(sample_z, [g_ema], args.log_dir,
                                       trigger=get_trigger((int(config['image_save_iter']), 'iteration')),
                                       plot_to_logger=True)
        trainer.extend(plotter)

    schedulers = {
        "generator": CosineAnnealingLR(generator_optimizer, int(config["max_iter"]), eta_min=1e-8),
        "discriminator": CosineAnnealingLR(discriminator_optimizer, int(config["max_iter"]), eta_min=1e-8),
    }
    lr_scheduler = LRScheduler(schedulers, trigger=get_trigger((1, 'iteration')))
    trainer.extend(lr_scheduler)

    trainer.extend(logger)

    synchronize()
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Stylegan 2 Generator")
    parser.add_argument("config", help="Path to yaml file holding train config")
    parser.add_argument("--resume-ckpt", default=None, type=str,
                        help="Path pretrained stylager to resume training from.")
    parser.add_argument("--images", required=True, help="path to json file holding a list of all images to use")
    parser.add_argument("--val-images",
                        help="path to json holding validation images (same data format as train images)")
    parser.add_argument("--device", choices=['cpu', 'cuda'], default='cuda', help='Device to use')
    parser.add_argument('-l', '--log-dir', default='training', help="outputs path")
    parser.add_argument('-ln', '--log-name', default='training', help='name of the train run')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mpi-backend', default='gloo', choices=['nccl', 'gloo'],
                        help="MPI backend to use for interprocess communication")
    parser.add_argument('--cache-root', help='path to local cache')
    parser.add_argument("-s", "--stylegan-variant", type=str.lower, choices=["1", "2", 'swagan'], default="2", help="which stylegan variant to use")
    parser.add_argument("--wandb-project-name", default="StyleGAN Training", help="The project name of the WandB project")  
    parser.add_argument("--wandb-entity", help="The name of the WandB entity")

    args = parser.parse_args()
    args.log_dir = os.path.join('logs', args.log_dir, args.log_name, datetime.datetime.now().isoformat())

    if torch.cuda.device_count() > 1:
        multiprocessing.set_start_method('forkserver')
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=args.mpi_backend, init_method='env://')
        synchronize()

    main(args, get_rank(), get_world_size())
