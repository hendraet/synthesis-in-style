import torch
from pytorch_training import Reporter
from pytorch_training.extensions import Evaluator
from torch.utils.data import Dataset

from networks.pixel_classifier.model import ensemble_eval_mode


class DiceGANEvalFunc:

    def __init__(self, ensemble, device: int, background_class_id: int = 0, num_classes: int = 3):
        self.ensemble = ensemble
        self.device = device
        self.background_class_id = background_class_id
        self.num_classes = num_classes

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), ensemble_eval_mode(self.ensemble):
            segmentation_result = self.ensemble.predict_classes(batch['images']).cpu()
        return segmentation_result


def calculate_dice_coefficient(pred: torch.Tensor, gt: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """
    computational formula： dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
    assert pred.shape == gt.shape, 'The shapes of prediction and groundtruth do not match.'
    N = gt.size(0)
    pred_flat = pred.view(N, -1).cuda()
    gt_flat = gt.view(N, -1).cuda()

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / N


class DatasetGANEvaluator(Evaluator):

    def __init__(self, *args, dataset: Dataset = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset

    def evaluate(self, reporter: Reporter):
        dice = 0
        num_batches = len(self.progress_bar())
        for batch_id, batch in enumerate(self.progress_bar()):
            images = torch.Tensor(batch['activations']).to(self.device)
            predictions = self.eval_func({'images': images}).squeeze().float()

            dice += calculate_dice_coefficient(predictions, torch.FloatTensor(batch["label"].float()))

        dice = dice / num_batches

        with reporter:
            log_key = "Dice Score"
            reporter.add_observation({log_key: dice}, prefix='evaluation')

        torch.cuda.empty_cache()
