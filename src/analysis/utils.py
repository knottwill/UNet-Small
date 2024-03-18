"""!@file utils.py

@brief Only contains function to construct a results dataframe from the predictions and ground truth masks
"""

import os
import numpy as np
import pandas as pd

from ..metrics import mask_accuracies, dice_coefficients


def construct_results_dataframe(dataroot, predictions_dir, train_cases, test_cases):
    """!
    @brief Construct a dataframe containing the results of the predictions for each slice of each case

    @details The dataframe contains the following columns:
    - Set: "train" or "test"
    - Case: case id
    - Slice Index: index of the slice (within the case arrays)
    - Accuracy: Accuracy of the prediction
    - DSC: Dice similarity coefficient of the prediction
    - Mask Size: proportion of mask pixels in the ground truth mask
    - Image: the image
    - Mask: the ground truth mask
    - Probabilities: the predicted probabilities

    @param dataroot: root directory of the LCTSC dataset
    @param predictions_dir: directory containing the predictions for each case
    @param train_cases: list of training case numbers
    @param test_cases: list of testing case numbers

    @return dataframe containing the results of the predictions
    """

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

    df = pd.DataFrame(data, columns=["Set", "Case", "Slice Index", "Accuracy", "DSC", "Mask Size", "Image", "Mask", "Probabilities"])

    return df
