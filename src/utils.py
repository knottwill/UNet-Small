"""!@file general.py

@brief General utility functions
"""
import torch


def slice_num(dicom):
    """!
    @brief Get the slice number from dicom filename

    >>> slice_num('1-041.dcm')
    41
    """
    assert isinstance(dicom, str), "dicom filename should be a string"
    return int(dicom.split("-")[1].split(".")[0])


def to_arrays(*items):
    """!
    Convert tensors shape (batch_size, 1, H, W) to numpy arrays of shape (batch_size, H, W)
    (unless batch_size=1, in which case we get numpy arrays of shape (H, W))
    """
    arrays = ()
    for item in items:
        if isinstance(item, torch.Tensor):
            assert len(item.shape) == 4 and item.shape[1] == 1, "Tensors should have the shape (batch_size, 1, H, W)"
            item = item.squeeze().detach().cpu().numpy()
        arrays += (item,)
    return arrays
