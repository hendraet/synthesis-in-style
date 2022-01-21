from pytorch_training import Updater
from pytorch_training.reporter import get_current_reporter
from pytorch_training.updater import GradientApplier
from torch import nn


class DatasetGANUpdater(Updater):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss()

    def update_core(self):
        batch = next(self.iterators['feature_vectors'])
        batch = {key: value.to(self.device) for key, value in batch.items()}
        reporter = get_current_reporter()

        for i, (key, network) in enumerate(self.networks.items()):
            with GradientApplier([network], [self.optimizers[f'optimizer_{i}']]):
                segmentation_prediction = network(batch['activations'])
                loss = self.loss(segmentation_prediction, batch['label'].long())

                loss.backward()
            reporter.add_observation({"CrossEntropyLoss_{}".format(key): loss}, 'loss')

    def reset(self):
        # make sure that the dataset is re-generated at every epoch if DatasetGANGenerationDataset is used
        for data_loader in self.data_loaders.values():
            try:
                data_loader.dataset.reset_dataset()
            except AttributeError:
                pass
        super().reset()
