import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def normalize(img: np.ndarray):
    '''Intensity value of the CT volumes is divided by 2048 and clamped between -1 and 1'''
    return np.clip(img / 2048, -1, 1)


def shift_scale_clamp(img: np.ndarray, shift=0.25, scale=0.25, clamp_min=-1, clamp_max=1) -> np.ndarray:
    '''For data augmentation during training,the intensity values are multiplied randomly with [0.75,1.25] and shifted by[-0.25,0.25].'''
    img += np.random.uniform(-shift, shift, img.shape)
    img *= 1+np.random.uniform(-scale, scale)
    img = np.clip(img, clamp_min, clamp_max)
    return img


def elastic_transformation(img: np.ndarray, mask: np.ndarray, alpha: float = 7, sigma: float = 3):
    '''Elastic deformation of images as described in [Simard2003]_.'''
    shape = img.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, 0, None, "reflect") * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, 0, None, "reflect") * alpha
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, 0, None, "reflect") * alpha

    z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    coordinates = np.reshape(z + dz, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    distorted_img = map_coordinates(img, coordinates, None, 3, 'reflect').reshape(shape)
    distorted_mask = map_coordinates(mask, coordinates, None, 1, 'reflect').reshape(shape)
    return distorted_img, distorted_mask
