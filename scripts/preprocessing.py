"""!@file preprocessing.py

@brief Script for pre-processing the LCTSC dataset

@details This script performs two main tasks:
- De-identifies the dicom files (removes patient name, ID, birth date and birth time)
- Converts them into numpy arrays per case (patient), aligned with the segmentation arrays,
and saves them in npz files.

Details about file orderings, to ensure the image arrays are aligned with the segmentations:
- For all cases EXCEPT Case_011 and Case_005, the dicom files have the naming convention
[1-001.dcm, 1-002.dcm, 1-003.dcm, ...], they are ordered from head to torso, and the segmentations
in the npz files are in opposite order to the images (eg. 1-002.dcm corresponds to masks[-2]).
- For Case_005, the naming convention is the same, but they are ordered from torso to head. Still,
the segmentations in the npz files are in opposite order to the images.
- For Case_011, the naming convention is [1-1.dcm, 1-2.dcm, ...] and the segmentations are already aligned with
the images. (eg. 1-001.dcm corresponds to masks[0], 1-002.dcm corresponds to masks[1], ...)

Usage:
Just need specify the root directory of the LCTSC dataset:

python scripts/preprocessing.py --dataroot ./Dataset
"""

import os
from os.path import join
import sys
import numpy as np
from pydicom import dcmread

# add project root to sys.path
project_root = os.path.abspath(join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.utils import slice_num
from src.arguments import parse_args

args = parse_args()

##################
# De-identification
##################

modified_values = []
for dir, subdirs, filenames in os.walk(join(args.dataroot, "Images")):  # walk through the dataset
    for filename in filenames:
        if filename.endswith(".dcm"):
            dcm = dcmread(join(dir, filename))  # read the dicom file
            case_id = os.path.basename(dir)

            modified = False

            # remove patient ID & replace with case_id
            if dcm["PatientID"].value != case_id:
                dcm["PatientID"].value = case_id
                modified_values.append(f"{case_id} - PatientID")
                modified = True

            # remove patient name & replace with case_id
            if dcm["PatientName"].value != case_id:
                dcm["PatientName"].value = case_id
                modified_values.append(f"{case_id} - PatientName")
                modified = True

            # remove patient birth date
            if dcm["PatientBirthDate"].value != "":
                dcm["PatientBirthDate"].value = ""
                modified_values.append(f"{case_id} - PatientBirthDate")
                modified = True

            # remove patient birth time
            if "PatientBirthTime" in dcm:
                del dcm["PatientBirthTime"]
                modified_values.append(f"{case_id} - PatientBirthTime")
                modified = True

            # save the modified dicom file
            if modified:
                dcm.save_as(join(dir, filename))

# print the modified values
modified_values = set(modified_values)
for value in modified_values:
    print("De-identified: ", value)
print("De-identification complete\n")

##################
# Conversion to numpy arrays per case
##################

# get cases
items = sorted(os.listdir(join(args.dataroot, "Images")))
cases = [item for item in items if os.path.isdir(join(args.dataroot, "Images", item))]

for case in cases:
    case_dir = join(args.dataroot, "Images", case)

    # get sorted list of dicom files in case directory, ordered in the
    # same order as the masks in the .npz files
    if case == "Case_011":
        case_files = sorted(os.listdir(case_dir), key=slice_num, reverse=False)
    else:
        case_files = sorted(os.listdir(case_dir), key=slice_num, reverse=True)

    # get array of images (CT scans)
    images = np.zeros((len(case_files), 512, 512))  # define array of zeros so that pixel_array is converted to float
    for i, file in enumerate(case_files):
        dcm = dcmread(join(case_dir, file))
        images[i] = dcm.pixel_array

    # save array in npz file
    filepath = join(args.dataroot, "Images", f"{case}.npz")
    np.savez(filepath, images=images)
    print("Saved: ", filepath)
