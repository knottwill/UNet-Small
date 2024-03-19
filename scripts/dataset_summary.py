"""!@file dataset_summary.py

@details This script is used to summarise the dataset. It prints the following information:
- Some of the key metadata for each case in the dataset (eg. Gender), from the DICOM files
- The number of training and testing images
- The proportion of empty masks in the training and testing sets
- The overall proportion of mask pixels in the dataset (out of all pixels)
- The proportion of mask pixels in the training and testing sets

Example usage:
python scripts/dataset_summary.py --dataroot ./Dataset
"""

import os
from os.path import join
import json
import sys
from pydicom import dcmread
import pandas as pd
import numpy as np

# add project root to sys.path
project_root = os.path.abspath(join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.arguments import parse_args

##################
# Case-level Metadata
##################

args = parse_args()  # parse command line arguments

# get metadata from DICOM files
cases = []
metadata = []
for dir, subdirs, filenames in os.walk(join(args.dataroot, "Images")):  # walk through the dataset
    for filename in filenames:
        if filename.endswith(".dcm"):  # if the file is a DICOM file
            dcm = dcmread(join(dir, filename))
            case_id = os.path.basename(dir)

            # if the case has already been added to the list, skip it
            if case_id in cases:
                continue

            # add metadata to the list
            cases.append(case_id)
            metadata.append(
                [
                    case_id,
                    dcm["Manufacturer"].value,
                    dcm["ManufacturerModelName"].value,
                    dcm["PatientSex"].value,
                    dcm["KVP"].value,
                    dcm["DistanceSourceToDetector"].value,
                    dcm["DistanceSourceToPatient"].value,
                    dcm["Exposure"].value,
                    dcm["PatientPosition"].value,
                ]
            )


# create a dataframe from the metadata
df = pd.DataFrame(
    metadata, columns=["Case_ID", "Manufacturer", "Manufacturer_Model", "Patient_S", "KVP", "Distance_Source_To_Detector", "Distance_Source_To_Patient", "Exposure", "Patient_Position"]
)

# sort by case_id
df = df.sort_values(by=["Case_ID"], ignore_index=True)

# print the metadata
print("\n++++++++++ Case Metadata ++++++++++++\n")
pd.set_option("display.max_columns", None)
print(df)


##################
# Train/Test Split Summary
##################

# load the train/test split
with open("train_test_split.json", "r") as f:
    split = json.load(f)
    train_cases = split["train"]
    test_cases = split["test"]

# load the masks from the train set
train_masks = None
for case in train_cases:
    filename = join("./Dataset", "Segmentations", f"{case}_seg.npz")
    masks = np.load(filename)["masks"]

    if train_masks is None:
        train_masks = masks
    else:
        train_masks = np.concatenate((train_masks, masks), axis=0)

# load the masks from the test set
test_masks = None
for case in test_cases:
    filename = join("./Dataset", "Segmentations", f"{case}_seg.npz")
    masks = np.load(filename)["masks"]

    if test_masks is None:
        test_masks = masks
    else:
        test_masks = np.concatenate((test_masks, masks), axis=0)

# number of images in the train/test sets
n_train_images = len(train_masks)
n_test_images = len(test_masks)

# proportion of the train/test sets which is made up of empty masks
empty_train_masks = np.sum(train_masks, axis=(1, 2)) == 0
empty_test_masks = np.sum(test_masks, axis=(1, 2)) == 0

empty_proportion_train = np.sum(empty_train_masks) / len(train_masks)
empty_proportion_test = np.sum(empty_test_masks) / len(test_masks)

# overall preportion of mask pixels in the dataset, train and test sets
mask_pixels = (np.sum(train_masks) + np.sum(test_masks)) / (len(train_masks.flatten()) + len(test_masks.flatten()))
train_mask_pixels = np.sum(train_masks) / len(train_masks.flatten())
test_mask_pixels = np.sum(test_masks) / len(test_masks.flatten())

# print the summary
print("\n++++++++++ Train/Test set summary ++++++++++++\n")
print(f"Train cases: {train_cases}")
print(f"Test cases: {test_cases}\n")
print(f"Number of training images: {n_train_images}")
print(f"Number of testing images: {n_test_images}\n")
print(f"Proportion of empty masks in the training set: {empty_proportion_train:.5f}")
print(f"Proportion of empty masks in the test set: {empty_proportion_test:.5f}\n")
print(f"Overall proportion of mask pixels in the dataset: {mask_pixels:.5f}")
print(f"Proportion of mask pixels in the training set: {train_mask_pixels:.5f}")
print(f"Proportion of mask pixels in the test set: {test_mask_pixels:.5f}")
