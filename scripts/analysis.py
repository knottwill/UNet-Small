"""!@file analysis.py

python scripts/analysis.py --dataroot ./Dataset --predictions_dir ./Predictions --metric_logger ./Models/metric_logger.pkl --output_dir ./plots
"""

import os
import sys

# add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import warnings

# ignore FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from src.arguments import parse_args
from src.analysis.utils import construct_results_dataframe
from src.analysis.visualisation import plot_metric_logger

args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# get results dataframe
df = construct_results_dataframe(args.dataroot, args.predictions_dir)
test_filter = df["Set"] == "test"
train_filter = df["Set"] == "train"

# First we split the dataframe into sets with and without ground-truth masks
test_no_mask = df[test_filter][df[test_filter]["Mask Size"] == 0]
test_with_mask = df[test_filter][df[test_filter]["Mask Size"] > 0]
train_no_mask = df[train_filter][df[train_filter]["Mask Size"] == 0]
train_with_mask = df[train_filter][df[train_filter]["Mask Size"] > 0]

empty_proportion_test = len(test_no_mask) / len(df[test_filter])
empty_proportion_train = len(train_no_mask) / len(df[train_filter])

###################
# Main Statistics
###################

print(f"Proportion of empty-mask images in test set: {empty_proportion_test:.5f}")
print(f"Proportion of empty-mask images in train set: {empty_proportion_train:.5f}\n")
print(f"Mean DSC for images with masks in test set: {test_with_mask['DSC'].mean():.5f}")
print(f"Mean DSC for images with masks in train set: {train_with_mask['DSC'].mean():.5f}\n")
print(f'Mean accuracy overall in test set: {df[test_filter]["Accuracy"].mean():.5f}')
print(f'Mean accuracy overall in train set: {df[train_filter]["Accuracy"].mean():.5f}\n')

print(f'Mean accuracy for empty masks in test set: {test_no_mask["Accuracy"].mean():.5f}')
empty_correctly_predicted = len(test_no_mask[test_no_mask["Accuracy"] == 1]) / len(test_no_mask)
print(f"Proportion of empty masks in test set with perfect accuracy: {empty_correctly_predicted:.5f}\n")

# calculate recall and specificity in test set
masks = np.stack(df[test_filter]["Mask"].to_numpy()).flatten()
probs = np.stack(df[test_filter]["Prediction"].to_numpy()).flatten()
pred = (probs > 0.5).astype(int)

recall = (masks * pred).sum() / masks.sum()
specificity = ((1 - masks) * (1 - pred)).sum() / (1 - masks).sum()

print(f"Recall in test set: {recall:.5f}")
print(f"Specificity in test set: {specificity:.5f}\n")

print(f"Proportion of positive (mask) pixels in test set: {(masks == 1).sum()/len(masks):.5f}")

################
# Plot loss, DSC and accuracy over training
################

with open(args.metric_logger, "rb") as f:
    metric_logger = pickle.load(f)

fig = plot_metric_logger(metric_logger, empty_proportion_test, empty_proportion_train)

fig.savefig(os.path.join(args.output_dir, "training-metrics.png"))

#################
# Histograms of DSC and Accuracy for test and train set
#################

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
sns.histplot(df[test_filter]["DSC"], bins=20, ax=ax[0, 0])
ax[0, 0].set_ylabel("Frequency", fontsize=16)
ax[0, 0].set_ylim(0, 415)
ax[0, 0].set_xlabel("DSC", fontsize=16)
ax[0, 0].legend(["test"], fontsize=16)
ax[0, 0].text(0.1, 0.9, "(a)", fontsize=16, transform=ax[0, 0].transAxes, fontweight="bold")

bins = np.linspace(0.96, 1.0, 21)
sns.histplot(df[test_filter]["Accuracy"], bins=bins, ax=ax[0, 1])
ax[0, 1].set_ylabel("")
ax[0, 1].set_xlabel("Accuracy", fontsize=16)
ax[0, 1].set_yticklabels([])
ax[0, 1].set_ylim(0, 415)
ax[0, 1].legend(["test"], fontsize=16)
ax[0, 1].set_xlim(0.96, 1.0)  # Setting limits for Accuracy
ax[0, 1].text(0.1, 0.9, "(b)", fontsize=16, transform=ax[0, 1].transAxes, fontweight="bold")

sns.histplot(df[train_filter]["DSC"], bins=20, ax=ax[1, 0], color="g")
ax[1, 0].set_ylabel("Frequency", fontsize=16)
ax[1, 0].set_ylim(0, 900)
ax[1, 0].set_xlabel("DSC", fontsize=16)
ax[1, 0].legend(["train"], fontsize=16)
ax[1, 0].text(0.1, 0.9, "(c)", fontsize=16, transform=ax[1, 0].transAxes, fontweight="bold")

bins = np.linspace(0.96, 1.0, 21)
sns.histplot(df[train_filter]["Accuracy"], bins=bins, ax=ax[1, 1], color="g")
ax[1, 1].set_ylabel("")
ax[1, 1].set_xlabel("Accuracy", fontsize=16)
ax[1, 1].set_yticklabels([])
ax[1, 1].set_ylim(0, 900)
ax[1, 1].legend(["train"], fontsize=16)
ax[1, 1].set_xlim(0.96, 1.0)  # Setting limits for Accuracy
ax[1, 1].text(0.1, 0.9, "(d)", fontsize=16, transform=ax[1, 1].transAxes, fontweight="bold")

plt.tight_layout()

fig.savefig(os.path.join(args.output_dir, "histograms.png"))


#####################
# Scatter plot of DSC vs Accuracy, colour indicates size of mask relative to image
#####################

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.scatterplot(x="DSC", y="Accuracy", hue="Mask Size", data=df[test_filter], palette="coolwarm", ax=ax, marker="o", s=70)
ax.set_title("DSC vs Accuracy")

fig.savefig(os.path.join(args.output_dir, "scatterplot.png"))


#########################
# Heatmaps: Slice on y-axis, case number on x-axis, colour represents:
# a) Mask Size
# b) DSC
# c) Accuracy
#########################

fig, ax = plt.subplots(1, 3, figsize=(19, 8))
sns.heatmap(df[test_filter].pivot(index="Slice Index", columns="Case", values="Mask Size"), cmap="viridis", ax=ax[0])
ax[0].set_ylabel("Slice Index", fontsize=16)
ax[0].invert_yaxis()
ax[0].set_xlabel("Case", fontsize=16)
ax[0].text(-0.05, 1.05, "(a)", fontsize=16, transform=ax[0].transAxes, fontweight="bold")
cbar = ax[0].collections[0].colorbar
cbar.set_label("Mask Size", fontsize=16)

sns.heatmap(df[test_filter].pivot(index="Slice Index", columns="Case", values="DSC"), cmap="viridis", ax=ax[1])
ax[1].invert_yaxis()
ax[1].set_yticklabels([])
ax[1].set_ylabel("")
ax[1].set_xlabel("Case", fontsize=16)
ax[1].text(-0.05, 1.05, "(b)", fontsize=16, transform=ax[1].transAxes, fontweight="bold")
cbar = ax[1].collections[0].colorbar
cbar.set_label("Dice Similarity Coefficient", fontsize=16)

sns.heatmap(df[test_filter].pivot(index="Slice Index", columns="Case", values="Accuracy"), cmap="viridis", ax=ax[2])
ax[2].invert_yaxis()
ax[2].set_yticklabels([])
ax[2].set_ylabel("")
ax[2].set_xlabel("Case", fontsize=16)
ax[2].text(-0.05, 1.05, "(c)", fontsize=16, transform=ax[2].transAxes, fontweight="bold")
cbar = ax[2].collections[0].colorbar
cbar.set_label("Accuracy", fontsize=16)

fig.savefig(os.path.join(args.output_dir, "heatmaps.png"))

###########
# Precision-Recall Curve
###########

print("Plotting precision-recall curve... (may take a few minutes)")
precision, recall, _ = precision_recall_curve(masks, probs)
AUC = auc(recall, precision)

fig, ax = plt.subplots()
ax.step(recall, precision, color="b", alpha=0.2, where="post")
ax.fill_between(recall, precision, step="post", alpha=0.2, color="b")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_ylim([0.0, 1.05])
ax.set_xlim([0.0, 1.05])
ax.set_title(f"AUC={AUC}")

# save figure
fig.savefig(os.path.join(args.output_dir, "precision_recall_curve.png"))