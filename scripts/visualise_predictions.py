"""!@file visualise_predictions.py

python scripts/visualise_predictions.py --dataroot ./Dataset --predictions_dir ./Predictions --output_dir ./plots
"""

import os
import sys

# add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import pandas as pd
import json

from src.arguments import parse_args
from src.analysis.utils import construct_results_dataframe
from src.analysis.visualisation import plot_rows

args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)

with open("train_test_split.json", "r") as f:
    split = json.load(f)
    train_cases = split["train"]
    test_cases = split["test"]

# get results dataframe
df = construct_results_dataframe(args.dataroot, args.predictions_dir, train_cases, test_cases)
test_filter = df["Set"] == "test"
train_filter = df["Set"] == "train"

# First we split the dataframe into sets with and without ground-truth masks
test_no_mask = df[test_filter][df[test_filter]["Mask Size"] == 0]
test_with_mask = df[test_filter][df[test_filter]["Mask Size"] > 0]
train_no_mask = df[train_filter][df[train_filter]["Mask Size"] == 0]
train_with_mask = df[train_filter][df[train_filter]["Mask Size"] > 0]

# best DSC
best = df[test_filter].sort_values("DSC", ascending=False).head(10)
fig = plot_rows(best.sample(3, random_state=42))
fig.savefig(os.path.join(args.output_dir, "best-dsc.png"))

# intermediate DSC
lowest_acc = test_with_mask.sort_values("Accuracy").head(2)
interval = test_with_mask[(test_with_mask["DSC"] > 0.0) & (test_with_mask["DSC"] < 0.6)].sample(2, random_state=0)
intermediate = pd.concat([lowest_acc, interval])

fig = plot_rows(intermediate)
fig.savefig(os.path.join(args.output_dir, "intermediate-dsc.png"))

# worst DSC
perf_no_mask = test_no_mask.sort_values("Accuracy", ascending=False).head(1)
worst_accuracy_no_mask = test_no_mask.sort_values("Accuracy").head(2)

# concatenate the two dataframes
no_mask = pd.concat([perf_no_mask, worst_accuracy_no_mask])

fig = plot_rows(no_mask)
fig.savefig(os.path.join(args.output_dir, "no-mask.png"))
