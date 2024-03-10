"""!@file general.py

@brief General utility functions
"""


def slice_num(dicom):
    """!
    @brief Get the slice number from dicom filename

    >>> slice_num('1-041.dcm')
    41
    """
    assert isinstance(dicom, str), "dicom filename should be a string"
    return int(dicom.split("-")[1].split(".")[0])
