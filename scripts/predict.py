"""!@file predict.py

@brief Script to predict on the LCTSC dataset using a pre-trained model

@details
Usage: Need to specify the following arguments:
- `--dataroot`: The root directory of the LCTSC dataset
- `--model_state_dict`: The path to the trained model state dict file
- `--output_dir`: The directory to save the predictions
- `--cases`: The cases to predict on. Can be 'all' or a list of case numbers (eg. '0,3,5' for Case_000, Case_003, Case_005)
- `--prediction_type`: The type of prediction to save. Must be 'prob' to re-produce the results from the report

Example:
`python scripts/predict.py --dataroot ./Dataset --model_state_dict ./Models/UNet_wdk24.pt --output_dir ./Predictions \
--cases all --prediction_type prob`
"""

import os
import sys

# add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.model import UNet
from src.data_loading import LCTDataset
from src.arguments import parse_args

args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

# load pre-trained model
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load(args.model_state_dict))

model.eval()
for case in args.cases:
    # create dataloader for the case
    case_dataset = LCTDataset(args.dataroot, [case])
    dataloader = DataLoader(case_dataset, batch_size=1, shuffle=False)

    # get probability predictions for all images in the case
    probs = torch.tensor([])
    for x, masks in tqdm(dataloader):
        x, masks = x.to(device), masks.to(device)
        pred = model(x).detach().cpu()
        probs = torch.cat((probs, pred), 0)

    # save the predictions
    save_path = os.path.join(args.output_dir, f"{case}_pred.npz")

    if args.prediction_type == "prob":  # MUST BE PROBABILITIES FOR THE ANALYSIS
        probs = probs.squeeze().detach().cpu().numpy()
        np.savez(save_path, probs=probs)
        print(f"Case {case} predicted probabilities saved to {save_path}")

    elif args.prediction_type == "mask":  # NOT USED FOR THE ANALYSIS
        masks_pred = (probs > 0.5).float()
        masks_pred = masks_pred.squeeze().detach().cpu().numpy()
        np.savez(save_path, masks=masks_pred)
        print(f"Case {case} predicted masks saved to {save_path}")
