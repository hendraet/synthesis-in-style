import torch
import torch.nn as nn


class LossFunction(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor: torch.Tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _metric_calculation(self, score, target):
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, weight=None, softmax=False) -> float:
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        loss = 0.0
        for i in range(self.n_classes):
            class_loss = self._metric_calculation(inputs[:, i], target[:, i])
            loss += class_loss * weight[i]
        return loss / self.n_classes


class DiceLoss(LossFunction):
    def _metric_calculation(self, score: torch.Tensor, target: torch.Tensor) -> float:
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss


class TverskyLoss(LossFunction):
    def __init__(self, n_classes: int, alpha: float, beta: float):
        super().__init__(n_classes)
        self.alpha = alpha
        self.beta = beta

    def _metric_calculation(self, score: torch.Tensor, target: torch.Tensor) -> float:
        target = target.float()
        smooth = 1e-5
        tp = torch.sum(score * target)
        fp = torch.sum((1-target) * score)
        fn = torch.sum(target * (1-score))
        loss = (tp + smooth) / (tp + self.alpha*fp + self.beta*fn + smooth)
        loss = 1 - loss
        return loss


class HandwritingLoss(LossFunction):
    def forward(self, inputs: torch.Tensor, target: torch.Tensor, weight=None, softmax: bool = False) -> float:
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        # This presumes that the handwritten class is always the last
        loss = self._metric_calculation(inputs[:, self.n_classes - 1], target[:, self.n_classes - 1])
        return loss


class Precision(LossFunction):
    def _metric_calculation(self, score: torch.Tensor, target: torch.Tensor) -> float:
        target = target.float()
        smooth = 1e-5
        tp = torch.sum(score * target)
        fp = torch.sum((1 - target) * score)
        return (tp + smooth) / (tp + fp + smooth)


class Recall(LossFunction):
    def _metric_calculation(self, score: torch.Tensor, target: torch.Tensor) -> float:
        target = target.float()
        smooth = 1e-5
        tp = torch.sum(score * target)
        fn = torch.sum(target * (1-score))
        return (tp + smooth) / (tp + fn + smooth)


class HandwritingPrecision(HandwritingLoss, Precision):
    def _metric_calculation(self, score: torch.Tensor, target: torch.Tensor) -> float:
        return super(HandwritingPrecision, self)._metric_calculation(score, target)

    def forward(self, inputs, target, weight=None, softmax=False) -> float:
        return super(HandwritingPrecision, self).forward(inputs, target, weight, softmax)


class HandwritingRecall(HandwritingLoss, Recall):
    def _metric_calculation(self, score: torch.Tensor, target: torch.Tensor) -> float:
        return super(HandwritingRecall, self)._metric_calculation(score, target)

    def forward(self, inputs, target, weight=None, softmax=False) -> float:
        return super(HandwritingRecall, self).forward(inputs, target, weight, softmax)

