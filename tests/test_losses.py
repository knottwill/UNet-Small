import os
import sys
import torch
from torch import nn

# add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.losses import ComboLoss, SoftDiceLoss


def test_combo_loss():
    probs = torch.sigmoid(torch.randn(2, 1, 512, 512))
    labels = torch.randint(0, 2, (2, 1, 512, 512)).float()

    loss = ComboLoss()
    loss(probs, labels)

    alpha = 2 / 3
    beta = 0.5

    combo_loss = ComboLoss()(probs, labels)

    # manual calculation of combo loss
    manual_combo_loss = alpha * beta * nn.BCELoss()(probs, labels) + (1 - alpha) * (SoftDiceLoss()(probs, labels) - 1.0)

    assert torch.isclose(combo_loss, manual_combo_loss)
