import argparse
import json
import os
from pathlib import Path

import numpy
from PIL import Image
from pytorch_training.images import make_image
from tqdm import tqdm

import global_config
from networks import load_autoencoder_or_generator
from utils.config import load_config
from utils.dataset_creation import get_base_dirs, build_latent_and_noise_generator, generate_images

if os.environ.get('REMOTE_PYCHARM_DEBUG_SESSION', False):
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=int(os.environ.get('REMOTE_PYCHARM_DEBUG_PORT')),
                            stdoutToServer=True, stderrToServer=True, suspend=False)


def extract_noise(batch, noise_files):
    noise_list = []
    for noise in batch.noise:
        noise = noise.cpu().numpy()
        noise_list.append(noise[0])
        key = f"noise_{(shape := noise.shape)[-2]}_{shape[-1]}"
        if key in noise_files:
            noise_files[key] = numpy.concatenate([noise_files[key], noise], axis=1)
        else:
            noise_files[key] = noise
    return noise_list


def main(args: argparse.Namespace):
    config = load_config(args.checkpoint, args.original_config_path)
    config['batch_size'] = args.batch_size
    image_save_base_dir = args.image_save_dir
    autoencoder = load_autoencoder_or_generator(args, config)

    data_loader = build_latent_and_noise_generator(autoencoder, config, seed=args.seed)
    data_iter = iter(data_loader)

    if args.truncate:
        mean_latent = autoencoder.decoder.mean_latent(4096)
    else:
        mean_latent = None

    data = []
    latent_codes = []
    save_activations = []
    with tqdm(total=args.num_images) as pbar:
        while pbar.n < args.num_images:
            batch = next(data_iter)
            activations, generated_images = generate_images(batch, autoencoder, mean_latent=mean_latent)

            generated_images = make_image(generated_images)

            num_generated_images = len(generated_images)
            assert len(batch.latent) == len(next(iter(activations.values()))) == num_generated_images, \
                "There has to be the same number of latents, activations, and generated images"
            image_path = image_save_base_dir
            for idx, init_vector in enumerate(batch.latent):
                image_id = pbar.n
                generated_image_path = f"generated_image_{image_id:03d}.png"
                generated_image = Image.fromarray(generated_images[idx])
                generated_image.save(Path(image_path, generated_image_path))

                generated_label_image_path = f"generated_image_label_{image_id:03d}.png"
                if args.generate_empty_label_images:
                    empty_label_image = Image.fromarray(numpy.zeros_like(generated_images[idx]))
                    empty_label_image.save(Path(image_path, generated_label_image_path))

                new_dataset_item = {
                    "image": generated_image_path,
                    "label": generated_label_image_path
                }

                if args.save_activations:
                    single_activations = {key: activations[key][idx].cpu().numpy() for key in activations.keys()}
                    save_activations.append(single_activations)
                    new_dataset_item["activations"] = len(save_activations) - 1

                if args.save_latents:
                    latent_codes.append(init_vector.cpu().numpy())
                    new_dataset_item["latent"] = len(latent_codes) - 1

                data.append(new_dataset_item)
                pbar.update(1)
                if not (pbar.n < args.num_images):
                    break

    if global_config.debug:
        # no need for gt if only creating debug images
        return

    with open(Path(image_save_base_dir, 'full_data.json'), 'w') as f:
        json.dump(data, f)

    split_index = int(len(data) * 0.8)
    with open(Path(image_save_base_dir, 'train.json'), 'w') as f:
        json.dump(data[:split_index], f)
    with open(Path(image_save_base_dir, 'test.json'), 'w') as f:
        json.dump(data[split_index:], f)

    print('Saving tensors...', end=' ')
    numpy.savez_compressed(Path(image_save_base_dir, 'tensors'), latent_codes=latent_codes,
                           activations=save_activations)
    print('Complete!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Creates a dataset based on StyleGAN, which can be used to train a "
                                     "DatasetGAN-like classifier")
    parser.add_argument("checkpoint", help="Path to trained autoencoder/generator for dataset creation")
    parser.add_argument("image_save_dir", type=Path, help="Where the images should be saved to")
    parser.add_argument("-op", "--original-config-path", type=Path, default=None,
                        help="Path to the YAML/JSON file that contains the config for the original segmenter "
                             "training Has to be provided if model was not trained and the original logging "
                             "structure is not present, i.e. the config does not lie in a sibling directory of the "
                             "checkpoint.")
    parser.add_argument("-n", "--num-images", type=int, default=100, help="Number of images to generate")
    parser.add_argument("-s", "--save-to",
                        help="path where to save generated images (default is save in dir of run of used checkpoint)")
    parser.add_argument("-b", "--batch-size", default=10, type=int, help="batch size for generation of images on GPU")
    parser.add_argument("-d", "--device", default='cuda',
                        help="CUDA device to use, either any (cuda) or the id of the device")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="render debug output during image generation")
    parser.add_argument("--truncate", action='store_true', default=False, help="Use truncation trick during generation")
    parser.add_argument("--save-latents", action='store_true', default=False, help="Save latent codes")
    parser.add_argument("--save-activations", action='store_true', default=False, help="Save avtications")
    parser.add_argument("-ge", "--generate-empty-label-images", action='store_true', default=False,
                        help="will create empty label images, but with the correct dimensions and file name")
    parser.add_argument("--seed", type=int, default=1, help="seed that should be used for the dataloader")

    parsed_args = parser.parse_args()
    if not (parsed_args.save_activations or parsed_args.save_latents):
        print("Neither --save-latents nor save-activations have been used. Thus, the resulting npz file will likely "
              "be empty.")
    if parsed_args.generate_empty_label_images:
        print("Empty label images will be created.")
    global_config.debug = parsed_args.debug
    main(parsed_args)
