import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self, reduction='sum', weight=1.):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, input, target):
        return self.weight * F.mse_loss(input, target, reduction=self.reduction)
