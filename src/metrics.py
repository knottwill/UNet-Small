import numpy as np


def mask_accuracies(preds, labels):
    """!
    @brief Calculate the accuracy of the predicted masks

    @param preds: predictions (masks or probabilities). Array of shape (batch_size, H, W)
    @param labels: true masks. Array of shape (batch_size, H, W)

    @return accuracies: array of shape (batch_size,)
    """
    assert preds.shape == labels.shape, "Predicted masks and true masks must have the same shape"
    assert len(preds.shape) == 3, "Predicted masks and true masks must have the shape (batch_size, H, W)"

    # convert to binary masks (if not already)
    preds = (preds > 0.5).astype(np.float32)

    # calculate accuracy
    accuracies = (preds == labels).mean(axis=(1, 2))

    return accuracies


def dice_coefficients(preds, labels):
    """!
    @brief Calculate the dice similarity coefficients of the predicted masks

    @param preds: predictions (masks or probabilities). Array of shape (batch_size, H, W)
    @param labels: true masks. Array of shape (batch_size, H, W)

    @return DSC: array of shape (batch_size,)
    """
    assert preds.shape == labels.shape, "Predicted masks and true masks must have the same shape"
    assert len(preds.shape) == 3, "Predicted masks and true masks must have the shape (batch_size, H, W)"

    # convert to binary masks (if not already)
    preds = (preds > 0.5).astype(np.float32)

    intersection = (preds * labels).sum(axis=(1, 2))
    union = preds.sum(axis=(1, 2)) + labels.sum(axis=(1, 2))

    dsc = (2.0 * intersection + 1e-7) / (union + 1e-7)

    assert dsc.shape == (preds.shape[0],), "Dice coefficients must have the shape (batch_size,)"

    return dsc
