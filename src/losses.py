import torch
from torch import nn


class ComboLoss(nn.Module):
    """!
    @brief Combination of soft-dice loss and modified cross entropy loss, as introduced in the paper
    "Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation"
    https://arxiv.org/pdf/1805.02798.pdf

    alpha is the weight of the modified cross entropy loss

    beta < 0.5 penalizes false positives more, while beta > 0.5 penalize false negatives more
    alpha is the weight of the modified cross entropy loss.

    alpha = 2/3, beta = 0.5 yields equal weight to the loss terms since then alpha * beta = (1 - alpha)

    The smoothing term serves several purposes:
    - prevents division by zero
    - allows a non-zero derivative when there is no ground truth mask
    - gives rise to a smoother loss surface which helps stabalize the learning process

    eps just a small constant to prevent numerical issues from log(probs)
    """

    def __init__(self, alpha=2 / 3, beta=0.5, smooth=1.0, eps=1e-7):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.eps = eps

    def forward(self, probs, labels):
        # calculate soft-dice coefficient
        intersection = (probs * labels).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + labels.sum() + self.smooth)

        # calculate modified cross entropy loss
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)
        modified_bce = -(self.beta * labels * torch.log(probs) + (1 - self.beta) * (1 - labels) * torch.log(1 - probs)).mean()

        # calculate combo loss
        combo = self.alpha * modified_bce - (1 - self.alpha) * dice

        return combo


class SoftDiceLoss(nn.Module):
    """!
    @brief soft-dice loss for binary segmentation

    The smoothing term can be set extremely small to avoid division by zero, but it can also be set to a
    larger value to ensure that the derivative of the loss function is not too harsh, which can help in a more
    stable and gradual learning process.

    """

    def __init__(self, p=1, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, probs, labels):
        """!
        @param probs: predicted probabilities - tensor of shape (batch_size, 1, H, W)
        @param labels: true masks - tensor of shape (batch_size, 1, H, W)

        @return loss: tensor of shape (1, )
        """

        intersection = (probs * labels).sum()
        denom = (probs.pow(self.p) + labels.pow(self.p)).sum()
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)

        return 1.0 - dice
