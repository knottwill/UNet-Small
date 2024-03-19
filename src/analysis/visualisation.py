"""!@file visualisation.py

@details This module contains functionality for visualising the results of the segmentation model.
It contains a function for plotting the images, masks, predictions and performance metrics for each row in the dataframe.
It contains a function for plotting the training and test loss, DSC and accuracy for each epoch.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..metrics import dice_coefficients, mask_accuracies


def plot_rows(df):
    """!
    @brief Plot the images, masks, predictions and performance metrics for each row in the dataframe.

    @details The dataframe should have the following columns:
    - Image: The input image.
    - Mask: The ground truth mask.
    - Probabilities: The predicted probabilities.
    - Case: The case id.
    - Slice Index: The slice index.

    @param df: The dataframe with the images, masks, predictions and metrics.

    @return fig: The figure with the images, masks, predictions and metrics.
    """
    fig, ax = plt.subplots(len(df), 4, figsize=(12, 3 * len(df)))
    plt.subplots_adjust(wspace=0, hspace=0)

    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    for i, (_, row) in enumerate(df.iterrows()):
        # Get the image, mask, prediction and metrics
        image = row["Image"]
        mask = row["Mask"]
        prediction = (row["Probabilities"] > 0.5).astype(np.uint8)  # Convert to binary mask
        case = row["Case"]
        slice_idx = row["Slice Index"]
        dsc = dice_coefficients(prediction, mask)  # Dice similarity coefficient
        accuracy = mask_accuracies(prediction, mask)  # Accuracy

        # Set titles just for the first row
        if i == 0:
            ax[i, 0].set_title("Input Image", fontsize=16)
            ax[i, 1].set_title("Ground Truth", fontsize=16)
            ax[i, 2].set_title("Prediction", fontsize=16)
            ax[i, 3].set_title("Metrics", fontsize=16)

        # Plot the image, mask and prediction
        ax[i, 0].imshow(image, cmap="gray")
        ax[i, 1].imshow(mask, cmap="gray")
        ax[i, 2].imshow(prediction, cmap="gray")

        # Put metrics in a text box in the last column
        ax[i, 3].text(0.4, 0.8, labels[i], fontsize=13, transform=ax[i, 3].transAxes, fontweight="bold")
        ax[i, 3].text(
            0.5,
            0.5,
            f"{case} slice {slice_idx}\n\nDSC: {dsc:.4f}\nAcc: {accuracy:.4f}",
            fontsize=13,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        )

        ax[i, 0].axis("off"), ax[i, 1].axis("off"), ax[i, 2].axis("off"), ax[i, 3].axis("off")
        ax[i, 3].set_aspect("equal")

    plt.tight_layout()

    return fig


def plot_metric_logger(metric_logger, empty_proportion_train, empty_proportion_test):
    """!
    @brief Plot the training and test loss, DSC and accuracy for each epoch.

    @details DSC is always zero for images with empty masks, hence the mean DSC is skewed
    by the number of empty masks. To counteract this, we scale the DSC values by the proportion
    of empty masks so that we get the mean DSC only for the images with masks. Specifically,
    we divide the DSC by (1 - empty_proportion) where empty_proportion is the proportion of empty masks.

    @param metric_logger: The dictionary with the training and test loss, DSC and accuracy.
    @param empty_proportion_train: The proportion of empty masks in the training set.
    @param empty_proportion_test: The proportion of empty masks in the test set.

    @return fig: The figure with the training and test loss, DSC and accuracy for each epoch.
    """

    # Get the number of epochs
    n_epochs = len(metric_logger["Loss"]["train"])
    epochs = np.arange(1, n_epochs + 1)

    # Loss plot
    fig, ax = plt.subplots(3, 1, figsize=(7, 10))
    ax[0].plot(epochs, metric_logger["Loss"]["train"], label="Train")

    if metric_logger["Loss"]["test"]:  # Only plot test loss if it exists
        ax[0].plot(epochs, metric_logger["Loss"]["test"], label="Test")

    ax[0].set_xlabel("")
    ax[0].set_xticklabels("")
    ax[0].set_xticks(epochs)
    ax[0].set_ylabel("Combo Loss", fontsize=16)
    ax[0].text(0.9, 0.5, "(a)", fontsize=16, transform=ax[0].transAxes, fontweight="bold")
    ax[0].legend()

    # DSC plot
    # Scale the DSC by the proportion of empty masks
    dsc_train = np.array(metric_logger["DSC"]["train"]) / (1 - empty_proportion_train)
    ax[1].plot(epochs, dsc_train, label="Train")

    if metric_logger["DSC"]["test"]:
        dsc_test = np.array(metric_logger["DSC"]["test"]) / (1 - empty_proportion_test)
        ax[1].plot(epochs, dsc_test, label="Test")

    ax[1].set_xlabel("")
    ax[1].set_xticklabels("")
    ax[1].set_xticks(epochs)
    ax[1].set_ylabel("DSC", fontsize=16)
    ax[1].text(0.9, 0.5, "(b)", fontsize=16, transform=ax[1].transAxes, fontweight="bold")
    ax[1].legend()

    # Accuracy plot
    ax[2].plot(epochs, metric_logger["Accuracy"]["train"], label="Train")

    if metric_logger["Accuracy"]["test"]:
        ax[2].plot(epochs, metric_logger["Accuracy"]["test"], label="Test")

    ax[2].set_xticks(epochs)
    ax[2].set_xlabel("Epoch", fontsize=16)
    ax[2].set_ylabel("Accuracy", fontsize=16)
    ax[2].text(0.9, 0.5, "(c)", fontsize=16, transform=ax[2].transAxes, fontweight="bold")
    ax[2].legend()

    plt.tight_layout()

    return fig
