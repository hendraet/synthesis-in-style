import torch
import torch.nn.functional as F

from losses.lpips import PerceptualLoss
from losses.perceptual_style_loss import StyleLossNetwork, PerceptualAndStyleLoss
from pytorch_training import Updater
from pytorch_training.reporter import get_current_reporter
from pytorch_training.updater import GradientApplier


class AdversarialAutoencoderUpdater(Updater):

    def __init__(self, *args, regularization_settings: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.regularization_settings = regularization_settings if regularization_settings is not None else {}
        self.perceptual_loss = PerceptualAndStyleLoss(use_style_loss=False).to(self.device)

    def update_core(self):
        reporter = get_current_reporter()
        binary_image = next(self.iterators['binary_images']).to(self.device)
        style_image = next(self.iterators['original_images']).to(self.device)
        discriminator_observations = self.update_discriminator(binary_image.clone().detach(), style_image.detach())
        reporter.add_observation(discriminator_observations, 'discriminator')

        generator_observations = self.update_generator(binary_image.detach(), style_image.detach())
        reporter.add_observation(generator_observations, 'generator')

    def update_discriminator(self, binary_image: torch.Tensor, style_image: torch.Tensor) -> dict:
        generator = self.networks['generator']
        discriminator = self.networks['discriminator']
        discriminator_optimizer = self.optimizers['discriminator']

        with GradientApplier(self.networks.values(), [discriminator_optimizer]):
            fake_color_image = generator(binary_image, style_image.detach())
            # fake_color_image = generator(binary_image)
            fake_prediction = discriminator(fake_color_image)
            fake_loss = F.softplus(fake_prediction).mean()
            fake_loss.backward()

            real_prediction = discriminator(style_image.detach())
            real_loss = F.softplus(-real_prediction).mean()
            real_loss.backward()

            discriminator_loss = real_loss.detach() + fake_loss.detach()

        loss_data = {
            'loss': discriminator_loss,
            'real_score': real_prediction.mean(),
            'fake_score': fake_prediction.mean()
        }

        if self.iteration % self.regularization_settings['d_interval'] == 0:
            style_image.requires_grad = True
            real_prediction = discriminator(style_image)
            grad_of_reference_image, = torch.autograd.grad(outputs=real_prediction.sum(), inputs=style_image, create_graph=True)
            gradient_penalty = grad_of_reference_image.pow(2).view(grad_of_reference_image.shape[0], -1).sum(1).mean()

            discriminator.zero_grad()
            (self.regularization_settings['r1_weight'] / 2 * gradient_penalty * self.regularization_settings['d_interval'] + 0 * real_prediction[0]).backward()
            discriminator_optimizer.step()

            loss_data['gradient_penalty'] = self.regularization_settings['r1_weight'] / 2 * gradient_penalty.detach().cpu() * self.regularization_settings['d_interval']

        torch.cuda.empty_cache()

        return loss_data

    def create_mask(self, image: torch.Tensor) -> torch.Tensor:
        mask_image = image.clone().detach()
        mask_image = mask_image[:, 0, ...].unsqueeze(1)
        mask_image = (mask_image + 1) / 2
        return 1 - mask_image

    def update_generator(self, binary_image: torch.Tensor, style_image: torch.Tensor) -> dict:
        generator = self.networks['generator']
        discriminator = self.networks['discriminator']
        reconstructor = self.networks['reconstructor']

        generator_optimizer = self.optimizers['generator']
        log_data = {}

        with GradientApplier(self.networks.values(), [generator_optimizer]):
            fake_color_image = generator(binary_image, style_image)

            reconstructed_binary_image = reconstructor(fake_color_image)

            mask_image = self.create_mask(binary_image)
            binary_reconstruction_loss = torch.square(mask_image * (reconstructed_binary_image - binary_image)).mean()
            # binary_reconstruction_loss = F.mse_loss(reconstructed_binary_image, binary_image)
            reconstruction_loss = self.loss_weights['reconstruction'] * binary_reconstruction_loss

            style_loss, perceptual_loss = self.perceptual_loss(fake_color_image, style_image, mask_image)
            style_loss = self.loss_weights['style'] * style_loss
            perceptual_loss = self.loss_weights['perceptual'] * perceptual_loss

            discriminator_prediction = discriminator(fake_color_image)
            discriminator_loss = self.loss_weights['discriminator'] * F.softplus(-discriminator_prediction).mean()

            loss = reconstruction_loss + discriminator_loss + perceptual_loss + style_loss
            loss.backward()

        log_data.update({
            "loss": loss,
            "discriminator_loss": discriminator_loss,
            "reconstruction_loss": binary_reconstruction_loss,
            "perceptual_loss": perceptual_loss,
            "style_loss": style_loss,
        })
        torch.cuda.empty_cache()

        return log_data
