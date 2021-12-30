import torch
import numpy as np
import SimpleITK as sitk
from typing import List


def resample(
    image: sitk.Image,
    method: int = sitk.sitkLinear,
    newSpacing: tuple = (1., 1., 1.)
) -> sitk.Image:
    '''Resample input image with given spacing'''
    size = image.GetSize()
    spacing = image.GetSpacing()
    newSize = [
        int(np.round(size[0]*(spacing[0]/newSpacing[0]))),
        int(np.round(size[1]*(spacing[1]/newSpacing[1]))),
        int(np.round(size[2]*(spacing[2]/newSpacing[2]))),
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(method)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetSize(newSize)
    resampler.SetOutputSpacing(newSpacing)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    return resampler.Execute(image)


def normalize(img: np.ndarray):
    '''Intensity value of the CT volumes is divided by 2048 and clamped between -1 and 1'''
    return np.clip(img / 2048, -1, 1)


class Aggregator(object):
    def __init__(self, channel: int = 1) -> None:
        super().__init__()
        self.channel = channel
        self.patches: List[torch.Tensor] = []
        self.location: List[int] = []

    def Add(self, pred: torch.Tensor, z: int):
        '''pred:(1,C,Z,Y,X)'''
        self.patches.append(pred)
        self.location.append(z)

    def Execute(self, mode: str = 'max') -> torch.Tensor:
        shape = (self.location[-1]+128, 96, 96)
        result = torch.zeros((len(self.patches), self.channel, *shape))
        for i, z in enumerate(self.location):
            result[i, :, z:z+128] = self.patches[i].cpu()
        if mode == 'max':
            result = torch.max(result, dim=0)[0]
        elif mode == 'average':
            result = torch.mean(result, dim=0)
        else:
            raise ValueError(f"Unsupported mode:{mode}")
        return result.unsqueeze(0)
