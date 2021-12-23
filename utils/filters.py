import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence, Dict, Type


CONV: Dict[int, Union[Type[F.conv1d], Type[F.conv2d], Type[F.conv3d]]] = {
    1: F.conv1d,
    2: F.conv2d,
    3: F.conv3d
}


class Gaussian(nn.Module):
    def __init__(
        self,
        dim: int = 2,
        kernel_size: Union[int, Sequence] = None,
        sigma: Union[float, Sequence] = 1.,
        groups: int = 1,
        trunc: float = 2.,
        mode: str = 'zeros',
        norm: bool = False,
    ) -> None:
        '''
        Pytorch implementation of GaussianBlur which can be used on GPU.
        dim: dimension.
        kernel_size: size of the gaussian kernel, if None, kernel_size will be calculated by trunc and sigma.
        sigma: standard deviation of gaussian kernel.
        group: channel of input. convolution will be calculated channel-wised.
        trunc:
            in Matlab, default value is set to 2.
            in scipy, default value is set to 4.
            in OpenCV, default value is set to 3 for unsigned char and 4 for other dtype.
        mode: mode for boundary handler,'zeros', 'reflect','replicate' or 'circular'.
        norm: if true, rescale data range to [0,1]
        '''
        super().__init__()
        self._check_param(dim, mode)
        self.dim = dim
        self.groups = groups
        self.mode = mode
        self.norm = norm
        if isinstance(sigma, (float, int)):
            sigma = [sigma] * dim
        if kernel_size is not None:
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size]*dim
        else:
            kernel_size = [int(trunc*s+0.5)*2+1 for s in sigma]
        self.padding = [ksize//2 for ksize in kernel_size]

        # calculate gaussian kernel
        meshgrid = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32) for size in kernel_size
            ]
        )
        kernel = 1
        for size, std, mg in zip(kernel_size, sigma, meshgrid):
            mean = (size-1)/2
            kernel *= torch.exp(-((mg-mean)/std)**2/2)

        # regist kernel
        kernel = kernel/kernel.sum()
        kernel = kernel.reshape(1, 1, *kernel_size)
        kernel = kernel.repeat(groups, *[1] * (kernel.dim() - 1))
        self.register_buffer('weight', kernel)

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mode != 'zeros':
            out = CONV[self.dim](F.pad(inputs, self.padding, self.mode),
                                 self.weight, padding=self.padding, groups=self.groups)
        else:
            out = CONV[self.dim](inputs, self.weight,
                                 padding=self.padding, groups=self.groups)
        return out/out.max() if self.norm else out

    def _check_param(self, dim, mode):
        if dim > 3:
            raise ValueError("dim > 3 is not supported!")
        if mode not in ['zeros', 'reflect', 'replicate', 'circular']:
            raise ValueError(f"Unsupported mode:{mode}")
