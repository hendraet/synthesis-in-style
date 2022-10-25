from torch import distributed


def reduce_sum(tensor):
    if not distributed.is_available():
        return tensor

    if not distributed.is_initialized():
        return tensor

    tensor = tensor.clone()
    distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)

    return tensor
