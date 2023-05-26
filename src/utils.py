import numpy as np

import pydicom as dicom
from pydicom.pixel_data_handlers import apply_modality_lut


def _min_max_normalize(image):
    """Normalize image to [0;1] range."""
    i_min, i_max = np.min(image), np.max(image)

    return (image - i_min) / (i_max - i_min + 1e-8)


def read_dicom(path, normalize=False):
    """Read dicom image."""
    data = dicom.dcmread(path)
    image = data.pixel_array

    image[image <= -2000] = 0
    image = apply_modality_lut(image, data)
    image = _min_max_normalize(image) if normalize else image

    return image.astype(np.float32)
