import argparse
import pickle
from collections import defaultdict
from itertools import count
from pathlib import Path
from typing import Union, Iterable, Tuple, Dict, Callable

import numpy
import torch
from PIL import ImageColor
from torch.nn.functional import interpolate
from torchvision.utils import save_image
from tqdm import trange

from create_dataset_for_segmentation import build_latent_and_noise_generator
from segmentation.gan_local_edit.factor_catalog import FactorCatalog
from latent_projecting import Latents
from networks import StyleganAutoencoder, TwoStemStyleganAutoencoder, load_autoencoder_or_generator
from pytorch_training.images import make_image
from utils.config import load_config
from utils.data_loading import build_data_loader

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


def get_next_color() -> tuple:
    while True:
        for color in COLOR_MAP:
            yield ImageColor.getrgb(color)


def get_next_class_color() -> tuple:
    for i in count():
        yield ImageColor.getrgb(f"#{i:06}")


def prepare_output_dir(args: argparse.Namespace) -> Path:
    root_dir = Path(args.checkpoint).parent.parent
    output_dir = root_dir / args.destination
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def get_activations(args: argparse.Namespace, autoencoder: Union[StyleganAutoencoder, TwoStemStyleganAutoencoder],
                    data_loader: Iterable) -> Tuple[Dict[int, torch.Tensor], numpy.ndarray]:
    all_activations = defaultdict(list)
    images = []
    for _ in range(0, args.num_samples, args.batch_size):
        batch = next(data_loader)

        if not isinstance(batch, Latents):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            latents = autoencoder.encode(batch['input_image'])
        else:
            latents = batch.to(args.device)

        with torch.no_grad():
            generated_image, intermediate_activations = autoencoder.decoder(
                [latents.latent],
                input_is_latent=False if isinstance(batch, Latents) else autoencoder.is_wplus(latents),
                noise=latents.noise,
                return_intermediate_activations=True
            )
            for key, activation in intermediate_activations.items():
                all_activations[key].append(activation.cpu())
            images.append(make_image(generated_image))

    all_activations = {key: torch.cat(value, dim=0) for key, value in all_activations.items()}
    images = numpy.concatenate(images, axis=0).transpose((0, 3, 1, 2))
    return all_activations, images


def strip_activations(activations: Dict[int, torch.Tensor], min_size: int) -> Dict[int, torch.Tensor]:
    return {key: value for key, value in activations.items() if (s := value.shape)[-2] > min_size and s[-1] > min_size}


def cluster_id_to_image(cluster_image: torch.Tensor, color_func: Callable) -> torch.Tensor:
    batch_size, n_clusters, height, width = cluster_image.shape
    output = torch.zeros((batch_size, 3, height, width), dtype=cluster_image.dtype)

    for cluster_id, color in zip(range(n_clusters), color_func()):
        mask = cluster_image[:, cluster_id, ...] == 1
        for i, color_component in enumerate(color):
            output_slice = output[:, i, ...]
            output_for_cluster = torch.where(mask, torch.full_like(output_slice, color_component), output_slice)
            output[:, i, ...] = output_for_cluster

    return output.view(-1, 3, height, width)


def find_and_render_clusters(all_activations: Dict[int, torch.Tensor], num_clusters: int,
                             color_func: Callable = get_next_color) -> Tuple[Dict[int, torch.Tensor], Dict[str, FactorCatalog]]:
    rendered_clusters = {}
    catalogs = {}
    id_to_size_map = {}
    for size, activations in all_activations.items():
        new_catalog = FactorCatalog(num_clusters, compute_labels=True)
        found_clusters = new_catalog.fit_predict(activations, raw=True).get()
        rendered_clusters[size] = cluster_id_to_image(found_clusters, color_func)
        catalogs[str(size)] = new_catalog
        size_key = f"{(s := found_clusters.shape)[-2]}x{s[-1]}"
        id_to_size_map[size] = size_key

    catalogs['id_to_size_map'] = id_to_size_map

    return rendered_clusters, catalogs


def save_catalogs(catalogs: Dict[str, FactorCatalog], num_clusters: int, dest_dir: Path):
    dest_path = dest_dir / f"{num_clusters}.pkl"
    dest_path = dest_path.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    with dest_path.open('wb') as f:
        pickle.dump(catalogs, f)


def save_cluster_visualizations(cluster_images: Dict[int, torch.Tensor], num_clusters: int, dest_dir: Path):
    array_path = dest_dir / 'cluster_arrays' / f"{num_clusters}.npz"
    array_path = array_path.resolve()
    array_path.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez_compressed(str(array_path), **{str(k): v.numpy() for k, v in cluster_images.items()})

    largest_size = max([tensor.shape[-1] for tensor in list(cluster_images.values())])
    clusters = []
    for key in cluster_images:
        rendered = cluster_images[key]
        clusters.append(interpolate(rendered, largest_size))

    cluster_images = torch.stack(clusters, dim=0)

    image_path = dest_dir / 'cluster_images' / f"{num_clusters}.png"
    image_path = image_path.resolve()
    image_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = cluster_images.shape[1]
    cluster_images = cluster_images.reshape((-1,) + cluster_images.shape[2:])

    save_image(cluster_images, image_path, nrow=num_samples, normalize=True, range=(0, 255))


def main(args: argparse.Namespace):
    output_dir = prepare_output_dir(args)

    config = load_config(args.checkpoint, args.original_config_path)
    config['batch_size'] = args.batch_size
    autoencoder = load_autoencoder_or_generator(args, config)

    if args.images is not None:
        data_loader = build_data_loader(args.images, config, config.get('absolute', False), shuffle_off=True)
    else:
        data_loader = build_latent_and_noise_generator(autoencoder, config)

    activations, generated_images = get_activations(args, autoencoder, iter(data_loader))

    if args.strip_activations_from is not None:
        activations = strip_activations(activations, args.strip_activations_from)

    for num_clusters in trange(*args.cluster_range):
        rendered_clusters, catalogs = find_and_render_clusters(activations, num_clusters=num_clusters,
                                                               color_func=get_next_color)
        save_catalogs(catalogs, num_clusters, output_dir.resolve() / 'catalogs')

        rendered_clusters[max(rendered_clusters.keys()) + 1] = torch.from_numpy(generated_images)
        save_cluster_visualizations(rendered_clusters, num_clusters, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use a trained generator to produce images and their according semantic segmentation map")
    parser.add_argument("checkpoint", help="Path to trained autoencoder")
    parser.add_argument("-op", "--original-config-path", type=Path, default=None,
                        help="Path to the YAML/JSON file that contains the config for the original segmenter "
                             "training Has to be provided if model was not trained and the original logging "
                             "structure is not present, i.e. the config does not lie in a sibling directory of the "
                             "checkpoint.")
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument("--destination", default='semantic_segmentation',
                        help='Where to save results. If relative path is given, will be saved relative to in the '
                             'second parent directory of the checkpoint (corresponds to the root of the original log '
                             'directory if self trained checkpoint is used).')
    parser.add_argument("-b", "--batch-size", default=10, type=int, help="batch size for generation of images on GPU")
    parser.add_argument("-n", "--num-samples", default=100, type=int,
                        help="number of samples for prediction of clusters")
    parser.add_argument("-c", "--cluster-range", nargs=2, default=[3, 24], type=int,
                        help="number of clusters to analyze")
    parser.add_argument("-i", "--images",
                        help='path to dir with images that shall be used as base images (only works with autoencoder checkpoint)')
    parser.add_argument("-s", "--strip-activations-from", type=int,
                        help="throw away all predictions that are smaller or equal to the given size")

    main(parser.parse_args())
