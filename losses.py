from torchvision.ops import sigmoid_focal_loss
import torch.nn as nn
import torch

#adapted and simplified from kornia.losses
class BinaryFocalLossWithLogits(nn.Module):

    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none') -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(input, target, self.alpha, self.gamma, self.reduction)