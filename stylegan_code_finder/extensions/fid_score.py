from pathlib import Path
from typing import Union

from pytorch_training import Extension, Trainer
from pytorch_training.distributed import get_rank, synchronize
from pytorch_training.reporter import get_current_reporter
from torch import nn
from torch.utils.data import DataLoader

from stylegan_code_finder.evaluation.fid import FID, GenerativeFID
from stylegan_code_finder.networks import StyleganAutoencoder


class FIDScore(Extension):

    def __init__(self, model: nn.Module, data_loader: DataLoader, *args, dataset_path: Union[str, Path] = None, device: str = 'cuda', **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(model, StyleganAutoencoder):
            self.fid_calculator = FID(device=device)
        else:
            self.fid_calculator = GenerativeFID(device=device, latent_size=model.style_dim, batch_image_key='image')
        self.model = model
        self.data_loader = data_loader
        self.dataset_path = dataset_path

    def initialize(self, trainer: 'Trainer'):
        self.run(trainer)

    def finalize(self, trainer: 'Trainer'):
        self.run(trainer)

    def run(self, trainer: Trainer):
        fid = self.fid_calculator(self.model, self.data_loader, self.dataset_path)
        synchronize()
        if get_rank() == 0:
            with get_current_reporter() as reporter:
                reporter.add_observation({"fid_score": fid}, "evaluation")
