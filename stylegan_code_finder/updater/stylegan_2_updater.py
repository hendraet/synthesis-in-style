import math
import random
from collections import Iterable
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch import autograd, nn

from distributed import reduce_sum
from networks.stylegan2 import Generator
from pytorch_training import Updater
from pytorch_training.distributed import get_world_size
from pytorch_training.reporter import get_current_reporter
from pytorch_training.updater import UpdateDisabler, GradientApplier


class Stylegan2Updater(Updater):

    def __init__(self, *args, latent_size: int = 512, style_mixing_prob: float = 0.9,
                 regularization_options: dict = None, g_ema: Generator = None,
                 freeze_stochastic_noise_layers: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if regularization_options is None:
            regularization_options = {}

        self.latent_size = latent_size
        self.style_mixing_prob = style_mixing_prob
        self.g_reg_batch_size_shrink_factor = 2
        self.mean_path_length = 0
        self.mean_path_length_avg = 0
        self.accumulation_decay = 0.5 ** (32 / (10 * 1000))

        if isinstance(freeze_stochastic_noise_layers, Iterable):
            self.stochastic_noise_layers_to_freeze = freeze_stochastic_noise_layers
        else:  # we either want to freeze all (True) or none (False)
            if freeze_stochastic_noise_layers:
                self.stochastic_noise_layers_to_freeze = [
                    int(name.split('_')[1]) for name, buffer in g_ema.noises.named_buffers()
                ]
            else:
                self.stochastic_noise_layers_to_freeze = []

        self.d_reg_interval = int(regularization_options.get('d_reg_interval', 16))
        self.g_reg_interval = int(regularization_options.get('g_reg_interval', 4))
        self.r1_weight = float(regularization_options.get('r1_weight', 10))
        self.path_reg_weight = float(regularization_options.get('path_reg_weight', 2))

        assert g_ema is not None, "For Training of Stylegan2 we need an accumulation generator!"
        self.g_ema = g_ema

    def accumulate(self, trained_model: Generator, decay=0.999):
        par1 = dict(self.g_ema.named_parameters())
        if isinstance(trained_model, DistributedDataParallel):
            trained_model = trained_model.module
        par2 = dict(trained_model.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

    def make_noise(self, batch_size: int, n_noise: int) -> Tuple[torch.Tensor]:
        if n_noise == 1:
            return tuple([torch.randn(batch_size, self.latent_size, device=self.device)])
        noises = torch.randn(n_noise, batch_size, self.latent_size, device=self.device)
        return noises.unbind(0)

    def make_stochastic_noise(self):
        noise_buffers = [buffer for name, buffer in self.g_ema.noises.named_buffers()]
        stochastic_noises = []
        for noise_id, buffer in enumerate(noise_buffers):
            if noise_id in self.stochastic_noise_layers_to_freeze:
                stochastic_noises.append(buffer)
            else:
                stochastic_noises.append(None)
        return stochastic_noises

    def mixing_styles(self, batch_size: int) -> Tuple[torch.Tensor]:
        if self.style_mixing_prob > 0 and random.random() < self.style_mixing_prob:
            return self.make_noise(batch_size, 2)
        else:
            return self.make_noise(batch_size, 1)

    def d_logistic_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def d_r1_loss(self, real_pred: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    def g_nonsaturating_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        loss = F.softplus(-fake_pred).mean()

        return loss

    def requires_grad(self, network: nn.Module, flag: bool):
        for parameter in network.parameters():
            parameter.requires_grad = flag

    def g_path_regularize(self, fake_img: torch.Tensor, latents: torch.Tensor, mean_path_length: torch.Tensor,
                          decay: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(fake_img) / math.sqrt(
            fake_img.shape[2] * fake_img.shape[3]
        )
        grad, = autograd.grad(
            outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
        )
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

        path_penalty = (path_lengths - path_mean).pow(2).mean()

        return path_penalty, path_mean.detach(), path_lengths

    def update_discriminator(self, images: torch.Tensor) -> dict:
        generator = self.networks['generator']
        discriminator = self.networks['discriminator']

        with UpdateDisabler(generator), GradientApplier([discriminator], [self.optimizers['discriminator']]):
            styles = self.mixing_styles(len(images))

            generated_image, _ = generator(styles, noise=self.make_stochastic_noise())
            fake_prediction = discriminator(generated_image)
            real_prediction = discriminator(images)

            d_loss = self.d_logistic_loss(real_prediction, fake_prediction)

            d_loss.backward()

        return {
            "discriminator_loss": d_loss.detach(),
            "real_score": real_prediction.mean().detach(),
            "fake_score": fake_prediction.mean().detach(),
        }

    def regularize_discriminator(self, images: torch.Tensor) -> dict:
        discriminator = self.networks['discriminator']

        with GradientApplier([discriminator], [self.optimizers['discriminator']]):
            images.requires_grad = True
            real_pred = discriminator(images)
            r1_loss = self.d_r1_loss(real_pred, images)

            (self.r1_weight / 2 * r1_loss * self.d_reg_interval + 0 * real_pred[0]).backward()

        return {
            "r1_loss": r1_loss.detach()
        }

    def update_generator(self, images: torch.Tensor) -> dict:
        generator = self.networks['generator']
        discriminator = self.networks['discriminator']

        with UpdateDisabler(discriminator), GradientApplier([generator], [self.optimizers['generator']]):
            styles = self.mixing_styles(len(images))
            fake_images, _ = generator(styles, noise=self.make_stochastic_noise())
            fake_prediction = discriminator(fake_images)

            g_loss = self.g_nonsaturating_loss(fake_prediction)

            g_loss.backward()

        return {
            "generator_loss": g_loss.detach()
        }

    def regularize_generator(self, images: torch.Tensor) -> dict:
        generator = self.networks['generator']

        with GradientApplier([generator], [self.optimizers['generator']]):
            path_batch_size = max(1, len(images) // self.g_reg_batch_size_shrink_factor)
            noise = self.mixing_styles(path_batch_size)
            fake_images, latents = generator(noise, return_latents=True, noise=self.make_stochastic_noise())

            path_loss, self.mean_path_length, path_lengths = self.g_path_regularize(
                fake_images, latents, self.mean_path_length
            )
            weighted_path_loss = self.path_reg_weight * self.g_reg_interval * path_loss

            if self.g_reg_batch_size_shrink_factor:
                weighted_path_loss += 0 * fake_images[0, 0, 0, 0]

            weighted_path_loss.backward()

            self.mean_path_length_avg = (
                    reduce_sum(self.mean_path_length).item() / get_world_size()
            )

        return {
            "perceputal_path_loss": path_loss.detach(),
            "perceptual_path_lengths": path_lengths.mean().detach()
        }

    def update_core(self):
        batch = next(self.iterators['images'])
        batch = {key: value.to(self.device) for key, value in batch.items()}
        reporter = get_current_reporter()

        discriminator_observations = self.update_discriminator(batch['image'])
        reporter.add_observation(discriminator_observations, 'discriminator')

        d_regularize = self.iteration % self.d_reg_interval == 0
        if d_regularize:
            d_regularize_observations = self.regularize_discriminator(batch['image'])
            reporter.add_observation(d_regularize_observations, 'discriminator')

        generator_observations = self.update_generator(batch['image'])
        reporter.add_observation(generator_observations, 'generator')

        g_regularize = self.iteration % self.g_reg_interval == 0
        if g_regularize:
            g_regularize_observations = self.regularize_generator(batch['image'])
            reporter.add_observation(g_regularize_observations, 'generator')

        self.accumulate(self.networks['generator'], self.accumulation_decay)
