import os
import torch
from torchvision.transforms import v2 as TorchVisionTrns
import numpy.ma as ma

from utils import str_to_bool

ENABLE_T1 = str_to_bool(os.getenv('ENABLE_T1'))
ENABLE_T1CE = str_to_bool(os.getenv('ENABLE_T1CE'))
ENABLE_T2 = str_to_bool(os.getenv('ENABLE_T2'))
ENABLE_FLAIR = str_to_bool(os.getenv('ENABLE_FLAIR'))
mean_mask = ma.array([0.32969433, 0.17604456, 0.18700259, 0.23908363],
                     mask=[ENABLE_T1, ENABLE_T1CE, ENABLE_T2, ENABLE_FLAIR])
a = mean_mask[mean_mask.mask].data

std_mask = ma.array([0.32621744, 0.178607, 0.20538321, 0.25072563],
                    mask=[ENABLE_T1, ENABLE_T1CE, ENABLE_T2, ENABLE_FLAIR])
b = std_mask[std_mask.mask].data


class Normalize3D(object):

    def __init__(self, lMean, lStd):
        self.mean = lMean
        self.std = lStd

    """Normalize a 3d numpy image by channel"""

    def __call__(self, image):
        tImg = (image - self.mean) / self.std

        return tImg


class ToTensor(object):
    """Convert image in sample to Tensors."""

    def __call__(self, image):
        tImg = torch.from_numpy(image).permute(3, 0, 1, 2)

        return tImg


oDataTrns = TorchVisionTrns.Compose([  # <! Chaining transformations
    Normalize3D(a, b),
    # <! Normalizes the Data (https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)
    ToTensor(),  # <! Convert to Tensor (4,128,160,125)
    TorchVisionTrns.ToDtype(torch.float32, scale=True),
])

oLblTrns = TorchVisionTrns.Compose([  # <! Chaining transformations
    Normalize3D(a, b),
    # <! Normalizes the Data (https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)
    ToTensor(),  # <! Convert to Tensor (4,128,160,125)
    TorchVisionTrns.ToDtype(torch.float32, scale=True),
])
