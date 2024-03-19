"""!@file train.py

@brief Script for training the UNet model for lung segmentation

@details
Usage: Need to specify the following arguments:
- `--dataroot`: The root directory of the LCTSC dataset
- `--output_dir`: The directory to save the trained model and the metric logger
- `--include_testing`: Whether to evaluate the model on the test set after each epoch (this is necessary to re-produce the results from the report)

python ./scripts/train.py --dataroot ./Dataset --output_dir ./Models --include_testing 1
"""

import os
import sys

# add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import pickle
import json
import numpy as np
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.backends import cudnn

from src.arguments import parse_args
from src.model import UNet
from src.data_loading import LCTDataset
from src.train_eval import train, evaluate

# fix as many seeds as possible for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

args = parse_args()  # parse command line arguments

os.makedirs(args.output_dir, exist_ok=True)  # create the output directory if it doesn't exist

# load the train-test split
with open("train_test_split.json", "r") as f:
    split = json.load(f)
    train_cases = split["train"]
    test_cases = split["test"]
print(f"Training on {len(train_cases)} cases: {train_cases}\nTesting on {len(test_cases)} cases: {test_cases}.")

# create the train and test datasets
train_set = LCTDataset(args.dataroot, train_cases)
test_set = LCTDataset(args.dataroot, test_cases)

# create the train and test data loaders
train_loader = DataLoader(train_set, batch_size=3, shuffle=True)
test_loader = DataLoader(test_set, batch_size=3, shuffle=False)

# Define the device (in order of priority: GPU, MPS, CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

# Define the model
model = UNet(in_channels=1, out_channels=1)
model.to(device)

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

metric_logger = {"Loss": {"train": [], "test": []}, "Accuracy": {"train": [], "test": []}, "DSC": {"train": [], "test": []}}
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch and log the metrics
    train_loss, train_acc, train_dsc = train(model, device, train_loader, optimizer)
    print(f"[train] Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, DSC: {train_dsc:.4f}")
    metric_logger["Loss"]["train"].append(train_loss)
    metric_logger["Accuracy"]["train"].append(train_acc)
    metric_logger["DSC"]["train"].append(train_dsc)

    # save model state after each epoch
    save_path = os.path.join(args.output_dir, "UNet.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # evaluate the model on the test set
    if args.include_testing:
        test_loss, test_acc, test_dsc = evaluate(model, device, test_loader)
        print(f"[test] Epoch {epoch+1}/{num_epochs} - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, DSC: {test_dsc:.4f}")
        metric_logger["Loss"]["test"].append(test_loss)
        metric_logger["Accuracy"]["test"].append(test_acc)
        metric_logger["DSC"]["test"].append(test_dsc)

# save the metric_logger (as a pkl file)
save_path = os.path.join(args.output_dir, "metric_logger.pkl")
with open(save_path, "wb") as f:
    pickle.dump(metric_logger, f)
print(f"Metric logger saved to {save_path}")
