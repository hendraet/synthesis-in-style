import torch
import torch.nn as nn
from typing import List


class LossFunction(nn.Module):
    def __init__(self, n_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def _one_hot_encoder(self, input_tensor: torch.Tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _metric_calculation(self, score, target):
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, weight: List[float] = None, softmax: bool = False,
                handwriting=False) -> float:
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1 / self.n_classes] * self.n_classes
        assert inputs.size() == target.size(), f'predict {inputs.size()} & target {target.size()} shape do not match'
        loss = 0.0
        if handwriting:
            loss = self._metric_calculation(inputs[:, self.n_classes - 1], target[:, self.n_classes - 1])
        else:
            for i in range(self.n_classes):
                class_loss = self._metric_calculation(inputs[:, i], target[:, i])
                loss += class_loss * weight[i]
        return loss


class DiceLoss(LossFunction):
    def _metric_calculation(self, score: torch.Tensor, target: torch.Tensor) -> float:
        target = target.float()
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)
        loss = 1 - loss
        return loss


class TverskyLoss(LossFunction):
    def __init__(self, n_classes: int, alpha: float, beta: float):
        super().__init__(n_classes)
        self.alpha = alpha
        self.beta = beta

    def _metric_calculation(self, score: torch.Tensor, target: torch.Tensor) -> float:
        target = target.float()
        tp = torch.sum(score * target)
        fp = torch.sum((1-target) * score)
        fn = torch.sum(target * (1-score))
        loss = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        loss = 1 - loss
        return loss


class Precision(LossFunction):
    def _metric_calculation(self, score: torch.Tensor, target: torch.Tensor) -> float:
        target = target.float()
        tp = torch.sum(score * target)
        fp = torch.sum((1 - target) * score)
        return (tp + self.smooth) / (tp + fp + self.smooth)


class Recall(LossFunction):
    def _metric_calculation(self, score: torch.Tensor, target: torch.Tensor) -> float:
        target = target.float()
        tp = torch.sum(score * target)
        fn = torch.sum(target * (1-score))
        return (tp + self.smooth) / (tp + fn + self.smooth)
