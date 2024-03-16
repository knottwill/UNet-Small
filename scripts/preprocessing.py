"""!@file preprocessing.py

@brief Script converts the DICOM dataset into numpy arrays per case (patient),
aligned with the segmentation arrays, and saves them in npz files.

@details For all cases EXCEPT Case_011, the dicom files have the naming convention
[1-001.dcm, 1-002.dcm, 1-003.dcm, ...], and the segmentations in the npz files are in opposite
order to the images (eg. 1-002.dcm corresponds to masks[-2]). However, for Case_011,
the naming convention is [1-1.dcm, 1-2.dcm, ...] and the segmentations are already aligned with
the images. (eg. 1-001.dcm corresponds to masks[0], 1-002.dcm corresponds to masks[1], ...)

python scripts/preprocessing.py --dataroot ./Dataset
"""

import os
import sys
import numpy as np
from pydicom import dcmread

# add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.utils import slice_num
from src.arguments import parse_args

args = parse_args()

# get cases
items = sorted(os.listdir(os.path.join(args.dataroot, "Images")))
cases = [item for item in items if os.path.isdir(os.path.join(args.dataroot, "Images", item))]

for case in cases:
    case_dir = os.path.join(args.dataroot, "Images", case)

    # get sorted list of dicom files in case directory, ordered in the
    # same order as the masks in the .npz files
    if case == "Case_011":
        case_files = sorted(os.listdir(case_dir), key=slice_num, reverse=False)
    else:
        case_files = sorted(os.listdir(case_dir), key=slice_num, reverse=True)

    # get array of images (CT scans)
    images = np.zeros((len(case_files), 512, 512))
    for i, file in enumerate(case_files):
        dcm = dcmread(os.path.join(case_dir, file))
        images[i] = dcm.pixel_array

    # save array in npz file
    filepath = os.path.join(args.dataroot, "Images", f"{case}.npz")
    np.savez(filepath, images=images)
    print("Saved: ", filepath)
