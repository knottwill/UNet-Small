import matplotlib.pyplot as plt
import numpy as np

from ..metrics import dice_coefficients, mask_accuracies


def plot_rows(df):
    fig, ax = plt.subplots(len(df), 4, figsize=(12, 3 * len(df)))
    plt.subplots_adjust(wspace=0, hspace=0)

    for i, (_, row) in enumerate(df.iterrows()):
        image = row["Image"]
        mask = row["Mask"]
        prediction = (row["Prediction"] > 0.5).astype(np.uint8)
        case = row["Case"]
        slice_idx = row["Slice Index"]
        dsc = dice_coefficients(prediction, mask)
        accuracy = mask_accuracies(prediction, mask)

        if i == 0:
            ax[i, 0].set_title("Input Image")
            ax[i, 1].set_title("Ground Truth")
            ax[i, 2].set_title("Prediction")
            ax[i, 3].set_title("Metrics")

        ax[i, 0].imshow(image, cmap="gray")
        ax[i, 1].imshow(mask, cmap="gray")
        ax[i, 2].imshow(prediction, cmap="gray")

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
