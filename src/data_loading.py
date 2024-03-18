"""!@file data_loading.py

@brief Components for loading the LCTSC dataset
"""

import os
import numpy as np

import torch
from torch.utils.data import Dataset


class LCTDataset(Dataset):
    """!
    @brief LCTDataset class for loading 2D slices from the dataset.
    """

    def __init__(self, dataroot, cases):
        self.dataroot = dataroot  # path to the dataset

        # load images and masks
        images, masks = torch.tensor([]), torch.tensor([])
        for case in cases:
            # load images and masks
            case_images = np.load(os.path.join(dataroot, "Images", f"{case}.npz"))["images"]
            case_masks = np.load(os.path.join(dataroot, "Segmentations", f"{case}_seg.npz"))["masks"]

            # convert to tensors
            case_images = torch.tensor(case_images, dtype=torch.float32).unsqueeze(1)
            case_masks = torch.tensor(case_masks, dtype=torch.float32).unsqueeze(1)

            # concatenate
            images = torch.cat((images, case_images))
            masks = torch.cat((masks, case_masks))

        assert images.shape == masks.shape, "Images and masks should have the same shape"

        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]

        return image, mask
