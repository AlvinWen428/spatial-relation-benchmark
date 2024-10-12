import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryCEWithLogitLoss(nn.Module):
    def __init__(self):
        super(BinaryCEWithLogitLoss, self).__init__()

    def forward(self, logits, label, weight=None, reduction="mean"):
        if isinstance(logits, torch.Tensor):
            return F.binary_cross_entropy_with_logits(logits, label.to(dtype=torch.float32), weight, reduction=reduction)
        elif isinstance(logits, tuple):
            assert all(isinstance(logit, torch.Tensor) for logit in logits)
            return sum([F.binary_cross_entropy_with_logits(logit, label.to(dtype=torch.float32), weight, reduction=reduction) for logit in logits])
