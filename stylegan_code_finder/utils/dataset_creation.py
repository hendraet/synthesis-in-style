import argparse
from pathlib import Path
from typing import Union, Dict, Tuple, Iterable, Optional

import torch

from latent_projecting import Latents
from networks import StyleganAutoencoder, TwoStemStyleganAutoencoder


def get_root_dir_of_checkpoint(checkpoint_file: Union[str, Path]) -> Path:
    if isinstance(checkpoint_file, str):
        checkpoint_file = Path(checkpoint_file)
    return checkpoint_file.parent.parent


def get_base_dirs(args: argparse.Namespace) -> Tuple[Path, Path]:
    if getattr(args, 'semantic_segmentation_base_dir', None) is None:
        base_dir = get_root_dir_of_checkpoint(args.checkpoint)
        semantic_segmentation_base_dir = base_dir / 'semantic_segmentation'
    else:
        semantic_segmentation_base_dir = args.semantic_segmentation_base_dir
        base_dir = semantic_segmentation_base_dir.parent
    if args.save_to is None:
        image_save_base_dir = base_dir / "generated_images"
    else:
        image_save_base_dir = Path(args.save_to)
    image_save_base_dir.mkdir(parents=True, exist_ok=True)
    return image_save_base_dir, semantic_segmentation_base_dir


def build_latent_and_noise_generator(autoencoder: StyleganAutoencoder, config: Dict, seed=1) -> Iterable:
    torch.random.manual_seed(seed)
    while True:
        latent_code = torch.randn(config['batch_size'], config['latent_size'])
        noise = autoencoder.decoder.make_noise()
        yield Latents(latent_code, noise)


def generate_images(batch: Union[Latents, dict], autoencoder: Union[StyleganAutoencoder, TwoStemStyleganAutoencoder],
                    device: str = 'cuda', mean_latent: torch.Tensor = None) \
        -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    if not isinstance(batch, Latents):
        batch = {k: v.to(device) for k, v in batch.items()}
        latents = autoencoder.encode(batch['input_image'])
    else:
        latents = batch.to(device)

    with torch.no_grad():
        generated_image, intermediate_activations = autoencoder.decoder(
            [latents.latent],
            input_is_latent=False if isinstance(batch, Latents) else autoencoder.is_wplus(latents),
            noise=latents.noise,
            return_intermediate_activations=True,
            truncation=0.7 if mean_latent is not None else 1,
            truncation_latent=mean_latent
        )
        return intermediate_activations, generated_image


