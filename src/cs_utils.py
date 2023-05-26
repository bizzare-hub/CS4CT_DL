import math

import torch
import torch.nn.functional as F
import torch.fft as fft


def construct_fourier_ramp_fitler(size, device):
    """Construct ramp filter in Fourier space."""
    n = torch.cat((torch.arange(1, size / 2 + 1, 2, dtype=torch.int32),
                   torch.arange(size / 2 - 1, 0, -2, dtype=torch.int32)))
    f = torch.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (math.pi * n) ** 2

    fourier_filter = 2 * torch.real(fft.fft(f))  # ramp filter

    return fourier_filter.to(device)


def filter_sinogram(image):
    """Differiantable filtering step in FBP algorithm.
      Input: (*, n_angles, resolution)
    """
    device = image.device
    image = image.clone()
    n_angles, size = image.size()[-2:]

    padded_size = max(64, int(2 ** math.ceil(math.log2(2 * size))))
    pad = padded_size - size
    
    image = F.pad(image.float(), (0, pad, 0, 0))

    f = construct_fourier_ramp_fitler(padded_size, device)
    image_ft = fft.fft(image) * f
    
    image_filtered = torch.real(fft.ifft(image_ft))
    image_filtered = image_filtered[..., :-pad] * (math.pi / (2 * n_angles))

    return image_filtered.type_as(image)
