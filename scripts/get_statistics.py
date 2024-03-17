"""!@file get_statistics.py

python scripts/get_statistics.py --dataroot ./Dataset --predictions_dir ./Predictions
"""

import os
import sys
import numpy as np
import json

# add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.arguments import parse_args
from src.analysis.utils import construct_results_dataframe

args = parse_args()

with open("train_test_split.json", "r") as f:
    data = json.load(f)
    train_cases = data["train"]
    test_cases = data["test"]

# get results dataframe
df = construct_results_dataframe(args.dataroot, args.predictions_dir, train_cases, test_cases)
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
