import argparse
import datetime
import functools
import multiprocessing
import os
from pathlib import Path

import torch.distributed
from pytorch_training.distributed import synchronize, get_rank, get_world_size
from pytorch_training.extensions import Snapshotter
from pytorch_training.extensions.logger import WandBLogger
from pytorch_training.extensions.lr_scheduler import LRScheduler
from pytorch_training.optimizer import GradientClipAdam
from pytorch_training.trainer import DistributedTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

from networks import load_weights, get_stylegan2_wplus_autoencoder, get_stylegan2_wplus_style_autoencoder  # TODO: check this erroneous import
from networks.stylegan2.model import Discriminator as Stylegan2Discriminator
from updater.adversarial_updater import AdversarialAutoencoderUpdater
from utils.config import load_config, load_yaml_config
from utils.data_loading import build_data_loader
from utils.style_image_plotter import StyleImagePlotter


def main(args, rank, world_size):
    config = load_config(args.autoencoder_checkpoint, None)
    if args.overwrite_config:
        config.update(load_yaml_config(args.config))

    real_data_loader = build_data_loader(args.original_images, config, args.absolute)
    fake_data_loader = build_data_loader(args.fake_images, config, args.absolute)

    stylegan_variant = config['stylegan_variant']
    if stylegan_variant == 2:
        generation_network_func = get_stylegan2_wplus_style_autoencoder
        reconstruction_network_func = get_stylegan2_wplus_autoencoder
        discriminator = Stylegan2Discriminator(config['image_size'])
        discriminator = discriminator.to(rank)
    else:
        RuntimeError("Stylegan Variant unknown")

    generation_autoencoder = generation_network_func(
        config['image_size'],
        config['latent_size'],
        config['input_dim'],
    )
    load_weights(generation_autoencoder.decoder, args.autoencoder_checkpoint, key='decoder', convert=True)
    generation_autoencoder = generation_autoencoder.to(rank)

    reconstruction_autoencoder = reconstruction_network_func(
        config['image_size'],
        config['latent_size'],
        config['input_dim'],
    )
    for param in reconstruction_autoencoder.parameters():
        param.requires_grad = False

    load_weights(reconstruction_autoencoder, args.autoencoder_checkpoint, key='autoencoder', convert=True)
    reconstruction_autoencoder = reconstruction_autoencoder.to(rank)

    optimizer_opts = {
        'betas': (config['beta1'], config['beta2']),
        'weight_decay': config['weight_decay'],
        'lr': config['lr'],
    }
    generator_optimizer = GradientClipAdam(
        generation_autoencoder.trainable_parameters(),
        **optimizer_opts
    )
    discriminator_optimizer = GradientClipAdam(
        discriminator.parameters(),
        **optimizer_opts
    )

    if world_size > 1:
        distributed = functools.partial(DDP, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False, output_device=rank)
        generation_autoencoder = distributed(generation_autoencoder)
        # reconstruction_autoencoder = distributed(reconstruction_autoencoder)
        discriminator = distributed(discriminator)

    networks = {
        'generator': generation_autoencoder,
        'discriminator': discriminator,
        'reconstructor': reconstruction_autoencoder,
    }

    updater = AdversarialAutoencoderUpdater(
        iterators={
            'original_images': real_data_loader,
            'binary_images': fake_data_loader,
        },
        networks=networks,
        optimizers={
            'generator': generator_optimizer,
            'discriminator': discriminator_optimizer,
        },
        device='cuda',
        copy_to_device=False,
        regularization_settings={
            'd_interval': 16,
            'r1_weight': 10,
        },
        loss_weights={
            'reconstruction': 2,
            'discriminator': 1,
            'style': 1e-7,
            'perceptual': 0.1,
        }
    )

    trainer = DistributedTrainer(
        config['max_iter'],
        updater
    )

    extensions = []
    if rank == 0:
        extensions.append(
            Snapshotter(
                {key: value.module if hasattr(value, 'module') else value for key, value in networks.items()},
                args.log_dir, interval=config['snapshot_save_iter']
            )
        )

        plot_images = []
        for i in range(config['display_size']):
            if hasattr(fake_data_loader.sampler, 'set_epoch'):
                fake_data_loader.sampler.set_epoch(i)
            plot_images.append(next(iter(fake_data_loader))[0])

        style_plot_images = []
        for i in range(config['display_size']):
            if hasattr(real_data_loader.sampler, 'set_epoch'):
                real_data_loader.sampler.set_epoch(i)
            style_plot_images.append(next(iter(real_data_loader))[0])

        extensions.append(StyleImagePlotter(
            plot_images,
            [generation_autoencoder, reconstruction_autoencoder],
            args.log_dir,
            config['image_save_iter'],
            style_images=style_plot_images,
            plot_to_logger=True)
        )

    extensions.append(
        LRScheduler(
            {
                "generator": CosineAnnealingLR(generator_optimizer, config["max_iter"], eta_min=1e-8),
                "discriminator": CosineAnnealingLR(generator_optimizer, config["max_iter"], eta_min=1e-8),
            },
            interval=1
        )
    )

    extensions.append(
        WandBLogger(
            args.log_dir,
            args,
            config,
            os.path.dirname(os.path.realpath(__file__)),
            interval=config['log_iter'],
            master=rank == 0,
            project_name="WPI Stylegan Generation Autoencoder",
            run_name=args.log_name,
        )
    )

    for extension in extensions:
        trainer.extend(extension)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model that is able to generate images that look like WPI samples based on our generated samples")
    parser.add_argument("config", help="path to yaml config to use for training")
    parser.add_argument("autoencoder_checkpoint", help="path to a pre-trained autoencoder that consists of a stylegan + encoder trained on real images")
    parser.add_argument("-o", "--original-images", required=True, help="path to json file holding a list of all images to use")
    parser.add_argument("-f", "--fake-images", required=True, help="path to json file holding a list of all images to use")
    parser.add_argument("--val-images", help="path to json holding validation images (same data format as train images)")
    parser.add_argument("--absolute", action='store_true', default=False, help="indicate that your json contains absolute paths")
    parser.add_argument("-d", "--device", default='cuda', help="GPU or CPU device to use")
    parser.add_argument('-l', '--log-dir', default='training', help="outputs path")
    parser.add_argument('-ln', '--log-name', default='training', help='name of the train run')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mpi-backend', default='gloo', choices=['nccl', 'gloo'], help="MPI backend to use for interprocess communication")
    parser.add_argument('--overwrite-config', action='store_true', default=False, help="overwrite the yaml config of saved run with config supplied by command line arguments")

    args = parser.parse_args()
    args.log_dir = str(Path('logs') / args.log_dir / args.log_name / datetime.datetime.now().isoformat())

    if torch.cuda.device_count() > 1:
        multiprocessing.set_start_method('fork')
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=args.mpi_backend, init_method='env://')
        synchronize()

    main(args, get_rank(), get_world_size())
