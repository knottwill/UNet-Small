import os
import numpy as np
import pandas as pd

from ..metrics import mask_accuracies, dice_coefficients


def construct_results_dataframe(dataroot, predictions_dir, train_cases, test_cases):
    # load the images, ground truth masks and mask probability predictions
    # and calculate DSC and accuracy for all slices for all cases

    data = []

    for case in test_cases:
        images = np.load(os.path.join(dataroot, "Images", f"{case}.npz"))["images"]
        masks = np.load(os.path.join(dataroot, "Segmentations", f"{case}_seg.npz"))["masks"]
        probs = np.load(os.path.join(predictions_dir, f"{case}_pred.npz"))["probs"]
        accuracies = mask_accuracies(probs, masks)
        dscs = dice_coefficients(probs, masks)
        mask_proportions = masks.sum(axis=(1, 2)) / (512 * 512)  # proportion of mask pixels in each slice

        for i in range(len(images)):
            data.append(["test", case, i, accuracies[i], dscs[i], mask_proportions[i], images[i], masks[i], probs[i]])

    # same for training cases
    for case in train_cases:
        images = np.load(os.path.join(dataroot, "Images", f"{case}.npz"))["images"]
        masks = np.load(os.path.join(dataroot, "Segmentations", f"{case}_seg.npz"))["masks"]
        probs = np.load(os.path.join(predictions_dir, f"{case}_pred.npz"))["probs"]
        accuracies = mask_accuracies(probs, masks)
        dscs = dice_coefficients(probs, masks)
        mask_proportions = masks.sum(axis=(1, 2)) / (512 * 512)  # proportion of mask pixels in each slice

        for i in range(len(images)):
            data.append(["train", case, i, accuracies[i], dscs[i], mask_proportions[i], images[i], masks[i], probs[i]])

    df = pd.DataFrame(data, columns=["Set", "Case", "Slice Index", "Accuracy", "DSC", "Mask Size", "Image", "Mask", "Prediction"])

    return df
