import os
import sys
import torch

# add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.unet import UNet


def test_unet():
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 512, 512)

    probs = model(x)
    assert probs.shape == (2, 1, 512, 512), f"Expected output shape (2, 1, 512, 512), but got {probs.shape}"
    assert (probs >= 0).all() and (probs <= 1).all(), "Output probabilities should be between 0 and 1"
