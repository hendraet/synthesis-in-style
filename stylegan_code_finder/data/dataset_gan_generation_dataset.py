import numpy
import torch
from tqdm import tqdm

from create_dataset_for_segmentation import generate_images
from data.base_dataset_gan_dataset import BaseDatasetGANDataset
from latent_projecting import Latents
from utils.segmentation_utils import segmentation_image_to_class_image


class DatasetGANGenerationDataset(BaseDatasetGANDataset):
    def __init__(self, *args, generator_model: torch.nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator_model

        self.load_tensors(["latent_codes"])

        self.load_data()
        if self.random_sampling:
            self.create_sampling_buckets()
        self.reset_dataset()
        self.feature_vector_length = len(self.pixel_activations[0][0][0])

    def reset_dataset(self):
        for idx, latent in enumerate(self.latents):
            noise = self.generator.decoder.make_noise()
            inputs = Latents(torch.Tensor(latent).unsqueeze(dim=0), noise)
            activations, generated_images = generate_images(inputs, self.generator)

            image_activations = []
            for key in activations.keys():
                upscaled_feature_maps = self.upsamplers[key](activations[key][0].unsqueeze(1)).cpu()
                upscaled_feature_maps = upscaled_feature_maps.squeeze().numpy()
                for feature_map in upscaled_feature_maps:
                    image_activations.append(feature_map)

            transposed_activations = numpy.transpose(image_activations, (1, 2, 0))
            if self.pixel_activations is None:
                self.pixel_activations = numpy.zeros((len(self.latents), *transposed_activations.shape),
                                                     dtype=numpy.float32)
            self.pixel_activations[idx] = transposed_activations

    def load_data(self):
        self.latents = []
        assert self.init_vectors is not None, "Latent codes/init vectors were not loaded properly"
        for idx, entry in tqdm(enumerate(self.json_data), desc="Loading data"):
            self.latents.append(self.init_vectors[entry["latent"]])
            self.image_paths.append(self.dataset_path / entry["image"])
            label_string = str(self.dataset_path / entry["label"])
            label_image = self.loader(label_string)
            label_array = segmentation_image_to_class_image(numpy.array(label_image), self.background_class_name,
                                                            self.class_to_color_map)
            label_array = self.class_image_to_tensor(label_array).type(torch.LongTensor)
            self.pixel_labels.append(label_array.squeeze().numpy())
        self.pixel_labels = numpy.array(self.pixel_labels)
