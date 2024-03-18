"""!@file metrics.py

@brief Functions for calculating metrics for segmentation

@details The metrics we use are:
- Mask accuracy: the proportion of pixels that are correctly classified
- Dice similarity coefficient: the overlap between the predicted and true masks
"""

import numpy as np


def mask_accuracies(preds, labels):
    """!
    @brief Calculate the accuracy of the predicted masks

    @param preds: predictions (masks or probabilities). Array of shape (batch_size, H, W) or (H, W)
    @param labels: true masks. Array of shape (batch_size, H, W) or (H, W)

    @return accuracies: array of shape (batch_size,) or scalar
    """
    assert preds.shape == labels.shape, "Predicted masks and true masks must have the same shape"

    # convert to binary masks (if not already)
    preds = (preds > 0.5).astype(np.float32)

    # calculate accuracy
    accuracies = (preds == labels).mean(axis=(-2, -1))

    return accuracies


def dice_coefficients(preds, labels):
    """!
    @brief Calculate the dice similarity coefficients of the predicted masks

    @param preds: predictions (masks or probabilities). Array of shape (batch_size, H, W) or (H, W)
    @param labels: true masks. Array of shape (batch_size, H, W) or (H, W)

    @return DSC: array of shape (batch_size,) or scalar
    """
    assert preds.shape == labels.shape, "Predicted masks and true masks must have the same shape"

    # convert to binary masks (if not already)
    preds = (preds > 0.5).astype(np.float32)

    intersection = (preds * labels).sum(axis=(-2, -1))
    union = preds.sum(axis=(-2, -1)) + labels.sum(axis=(-2, -1))

    dsc = (2.0 * intersection) / (union + 1e-7)

    return dsc
