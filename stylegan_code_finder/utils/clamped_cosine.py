from pytorch_training.extensions.lr_scheduler import LRScheduler
from pytorch_training.triggers import get_trigger
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


class ClampedCosineAnnealingLR(CosineAnnealingLR):
    """
    Scheduler will only run until the number of epochs specified by cosine_max_update and afterwards return end_lr
    """
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
        super().__init__(optimizer, T_max, eta_min, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch > self.T_max:
            return [self.eta_min for _ in self.optimizer.param_groups]
        else:
            return super(ClampedCosineAnnealingLR, self).get_lr()


if __name__ == '__main__':
    segmentation_network = nn.Linear(10, 10)
    optimizer_opts = {
        'betas': (0.5, 0.99),
        'lr': 0.001,
    }
    optimizer = Adam(segmentation_network.parameters(), **optimizer_opts)
    schedulers = dict(encoder=ClampedCosineAnnealingLR(optimizer, 10, eta_min=0.00001))
    lr_scheduler = LRScheduler(schedulers, trigger=get_trigger((1, 'iteration')))
