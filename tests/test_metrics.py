import os
import sys
import numpy as np
import torch
from pytest import approx
from torchmetrics.classification import BinaryAccuracy, Dice

# add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.metrics import dice_coefficients, mask_accuracies


def test_metrics():
    probs = torch.rand(10, 1, 512, 512)
    labels = torch.randint(0, 2, (10, 1, 512, 512))

    accuracy = BinaryAccuracy(threshold=0.5, multidim_average="samplewise")(probs, labels).numpy()
    dice = Dice(zero_division=1, threshold=0.5, average="samples")(probs, labels).item()

    probs, labels = probs.squeeze().numpy(), labels.squeeze().numpy()

    assert (mask_accuracies(probs, labels) == accuracy).all(), f"Expected {accuracy}, got {mask_accuracies(probs, labels)}"
    assert dice_coefficients(probs, labels).mean() == approx(dice), f"Expected {dice}, got {dice_coefficients(probs, labels).mean()}"


def test_special_case():
    """Tests metrics when there are no masks in the predictions or labels"""
    probs = np.zeros((10, 512, 512))
    labels = np.zeros((10, 512, 512))

    assert (mask_accuracies(probs, labels) == 1.0).all()
    assert (dice_coefficients(probs, labels) == 1.0).all()
