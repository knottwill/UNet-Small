"""!@file visualise_predictions.py

@brief This script is used to generate statistics and visualisations of the predictions made by the model.

@details For the analysis of the model's predictions, it is necessary to separate the examples which
have non-zero ground-truth masks, and those which do not (since the DSC is always zero for the latter).

The script first calculates the main statistics of the predictions, such as the
mean DSC and accuracy for the test and train sets, as well as the overall recall and specificity.

It then generates visualisations of examples from the test set, from various catagories. For each of the
catagories, it calculates the proportion of the test set that falls into that catagory.
For the examples WITH non-zero ground-truth masks, the catagories are:
- The best predictions (DSC > 0.975)
- Intermediate predictions (DSC in the range (0.8, 0.975])
- The worst predictions (DSC < 0.8)
- The predictions with the lowest accuracy (specifically, these are examples from Case 011 that have particularly low accuracy - they are discussed in the report)

For examples WITHOUT non-zero ground-truth masks, the catagories are:
- The predictions with perfect accuracy
- The predictions with accuracy in the range (0.999, 1)
- The predictions with accuracy < 0.999

Usage: The script takes the following arguments:
- `--dataroot`: The root directory of the dataset
- `--predictions_dir`: The directory containing the predictions made by the model (should be probabilities)
- `--output_dir`: The directory where the visualisations will be saved

Example:
`python scripts/stats_and_visualisation.py --dataroot ./Dataset --predictions_dir ./Predictions --output_dir ./plots`
"""

import os
import sys

# add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import pandas as pd
import json
import numpy as np

from src.arguments import parse_args
from src.analysis.utils import construct_results_dataframe
from src.analysis.visualisation import plot_rows

# parse arguments and create output directory
args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# load train-test split
with open("train_test_split.json", "r") as f:
    split = json.load(f)
    train_cases = split["train"]
    test_cases = split["test"]

# get results dataframe
df = construct_results_dataframe(args.dataroot, args.predictions_dir, train_cases, test_cases)

# Filters for test and train sets
test_filter = df["Set"] == "test"
train_filter = df["Set"] == "train"

###################
# Main Statistics
###################

# Split the dataframe into sets with and without ground-truth masks
test_no_mask = df[test_filter][df[test_filter]["Mask Size"] == 0]
test_with_mask = df[test_filter][df[test_filter]["Mask Size"] > 0]
train_no_mask = df[train_filter][df[train_filter]["Mask Size"] == 0]
train_with_mask = df[train_filter][df[train_filter]["Mask Size"] > 0]

print("\n###########################################################")
print("##################### Main Statistics #####################\n")

print(f"Mean DSC for images with masks in test set: {test_with_mask['DSC'].mean():.5f}")
print(f"Mean DSC for images with masks in train set: {train_with_mask['DSC'].mean():.5f}\n")
print(f'Mean accuracy overall in test set: {df[test_filter]["Accuracy"].mean():.5f}')
print(f'Mean accuracy overall in train set: {df[train_filter]["Accuracy"].mean():.5f}\n')

print(f'Mean accuracy for empty masks in test set: {test_no_mask["Accuracy"].mean():.5f}')

empty_correctly_predicted = len(test_no_mask[test_no_mask["Accuracy"] == 1]) / len(test_no_mask)
print(f"Proportion of empty masks in test set with perfect accuracy: {empty_correctly_predicted:.5f}\n")

# calculate recall and specificity in test set
masks = np.stack(df[test_filter]["Mask"].to_numpy()).flatten()
probs = np.stack(df[test_filter]["Probabilities"].to_numpy()).flatten()
pred = (probs > 0.5).astype(int)

recall = (masks * pred).sum() / masks.sum()
specificity = ((1 - masks) * (1 - pred)).sum() / (1 - masks).sum()

print(f"Recall in test set: {recall:.5f}")
print(f"Specificity in test set: {specificity:.5f}\n")
print("###########################################################\n")

#####################
# Test-set examples with non-zero masks
#####################

# high, intermediate and low DSC categories
high_dsc = test_with_mask["DSC"] > 0.975
mid_dsc = (test_with_mask["DSC"] > 0.8) & (test_with_mask["DSC"] <= 0.975)
low_dsc = test_with_mask["DSC"] <= 0.8

print("Proportion of test set with DSC > 0.975: ", len(test_with_mask[high_dsc]) / len(df[test_filter]))
print("Proportion of test set with DSC in (0.8, 0.975]: ", len(test_with_mask[mid_dsc]) / len(df[test_filter]))
print("Proportion of test set with DSC < 0.8: ", len(test_with_mask[low_dsc]) / len(df[test_filter]))

# high dsc catagory
high = test_with_mask[high_dsc].sample(3, random_state=0)
fig = plot_rows(high)
fig.savefig(os.path.join(args.output_dir, "high-dsc.png"))

# intermediate dsc category
# we pick them out at regular intervals of DSC
intermediates = test_with_mask[mid_dsc]
intermediates = intermediates.sort_values("DSC", ascending=False)[12::30]
fig = plot_rows(intermediates)
fig.savefig(os.path.join(args.output_dir, "intermediate-dsc.png"))

# low dsc category
low = test_with_mask[low_dsc].sample(3, random_state=1)
fig = plot_rows(low)
fig.savefig(os.path.join(args.output_dir, "low-dsc.png"))

# lowest accuracy
worst = test_with_mask.sort_values("Accuracy", ascending=True).head(3)
fig = plot_rows(worst)
fig.savefig(os.path.join(args.output_dir, "worst-accuracy.png"))

#########################
# Test-set examples with no masks
# (DSC is zero always - accuracy is the only meaningful metric)
#########################

# perfect, high and low accuracy categories
perf_acc = test_no_mask["Accuracy"] == 1
high_acc = (test_no_mask["Accuracy"] >= 0.999) & (test_no_mask["Accuracy"] < 1)
low_acc = test_no_mask["Accuracy"] <= 0.999

print(f"Proportion of test-set empty-masks with perfect accuracy: {perf_acc.mean():.5f}")
print(f"Proportion of test-set empty-masks with accuracy in (0.999, 1): {high_acc.mean():.5f}")
print(f"Proportion of test-set empty-masks with accuracy < 0.999: {low_acc.mean():.5f}")

# get one example from each catagory
perf = test_no_mask[perf_acc].sample(1, random_state=0)
mid = test_no_mask[high_acc].sample(1, random_state=0)
worst = test_no_mask.sort_values("Accuracy", ascending=True).head(1)

# plot examples
no_mask = pd.concat([perf, mid, worst])
fig = plot_rows(no_mask)
fig.savefig(os.path.join(args.output_dir, "no-mask.png"))

# proportion of 'excellent' predictions in the test set
high_dsc = df[test_filter]["DSC"] > 0.975
high_acc_no_mask = (df[test_filter]["Mask Size"] == 0) & (df[test_filter]["Accuracy"] > 0.999)
excellent = df[test_filter][high_dsc | high_acc_no_mask]
print(f"\n\nProportion of test set with DSC > 0.975 or empty masks & accuracy > 0.999: {len(excellent) / len(df[test_filter])}")
