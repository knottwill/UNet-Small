import os
import sys
import numpy as np
from pytest import approx

# add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.metrics import dice_coefficients, mask_accuracies


probs = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
labels = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]])


def test_dice_coefficients():
    assert dice_coefficients(probs, labels).mean() == approx(0.85714, rel=1e-4)


def test_accuracy():
    assert mask_accuracies(probs, labels).mean() == approx(0.88889, rel=1e-4)


def test_special_case():
    """Tests metrics when there are no masks in the predictions or labels"""
    probs = np.zeros((10, 512, 512))
    labels = np.zeros((10, 512, 512))

    assert (mask_accuracies(probs, labels) == 1.0).all()
    assert (dice_coefficients(probs, labels) == 0.0).all()
