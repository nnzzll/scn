import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1e-5
    ) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        intersection = torch.sum(target*pred)
        denominator = target.sum() + pred.sum()
        loss: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth) / (denominator+self.smooth)
        return loss
