import torch
import random
import numpy as np
from typing import Sequence, List, Tuple
from torch.utils.data import TensorDataset


def random_sample(image: np.ndarray, mask: np.ndarray) -> Sequence[np.ndarray]:
    '''Crop a [128,96,96] patch'''
    if mask.shape == (128, 96, 96):
        return image, mask
    new_mask = np.zeros((128, 96, 96), dtype=np.uint8)
    classes = np.unique(new_mask)
    while(len(classes) == 1):
        # a little different to the paper
        z_range = (64, mask.shape[0]-64)
        z = random.randint(*z_range)
        new_mask = mask[z-64:z+64]
        classes = np.unique(new_mask)
        # in VerSe Dataset, vertebrae has no label when it is incomplete
        if len(classes) > 1:
            if (new_mask == classes[1]).sum() != (mask == classes[1]).sum():
                new_mask[new_mask == classes[1]] = 0
                # print("delete top",classes[1])
            if (new_mask == classes[-1]).sum() != (mask == classes[-1]).sum():
                new_mask[new_mask == classes[-1]] = 0
                # print("delete bottom",classes[-1])
        classes = np.unique(new_mask)
    new_image = image[z-64:z+64]
    return new_image, new_mask


def generate_landmark(mask: np.ndarray, num: int) -> np.ndarray:
    '''Generate a one-hot landmark volume'''
    landmark = np.zeros((num, *mask.shape), np.float32)
    classes = np.unique(mask)
    for i in range(1, num+1):
        if i in classes:
            Z, Y, X = np.where(mask == i)
            Z = np.round(Z.mean()).astype(int)
            Y = np.round(Y.mean()).astype(int)
            X = np.round(X.mean()).astype(int)
            landmark[i-1, Z, Y, X] = 1
    return landmark


def generate_one_channel_landmark(mask: np.ndarray) -> np.ndarray:
    '''Generate a single channel landmark volume'''
    landmark = np.zeros(mask.shape, np.float32)
    classes = np.unique(mask)
    for i in classes[1:]:
        Z, Y, X = np.where(mask == i)
        Z = np.round(Z.mean()).astype(int)
        Y = np.round(Y.mean()).astype(int)
        X = np.round(X.mean()).astype(int)
        landmark[Z, Y, X] = 1
    return landmark


def generate_patches(image: torch.Tensor, mask: torch.Tensor, num: int, overlap: int) -> Tuple[TensorDataset, List[int]]:
    '''Generate patch and correspoding location for inference.Batch size should be 1
    image: whole image volume [1,1,Z,96,96], where Z >= 128
    mask:  whole mask volume [1,1,Z,96,96], where Z >= 128
    num: number of classes
    overlap: overlapping area between patches will larger than this param
    '''
    image = image.squeeze().numpy()
    mask = mask.squeeze().numpy()
    dz = 128 - overlap
    z_range = list(range(0, mask.shape[0]-128, dz))
    z_range.append(mask.shape[0]-128)
    z_range = sorted(list(set(z_range)))  # remove duplicate
    location = []
    new_mask = []
    new_image = []
    new_landmark = []
    for z in z_range:
        temp = mask[z:z+128]
        classes = np.unique(temp)
        if len(classes) > 1:
            if (temp == classes[1]).sum() != (mask == classes[1]).sum():
                temp[temp == classes[1]] = 0
            if (temp == classes[-1]).sum() != (mask == classes[-1]).sum():
                temp[temp == classes[-1]] = 0
            if temp.sum():
                new_mask.append(np.expand_dims(temp, 0))
                new_image.append(np.expand_dims(image[z:z+128], 0))
                new_landmark.append(np.expand_dims(
                    generate_landmark(temp, num)[0], 0))
                location.append(z)
    new_image = torch.Tensor(np.vstack(new_image)).unsqueeze(1)  # [N,1,128,96,96]
    new_mask = torch.Tensor(np.vstack(new_mask)).unsqueeze(1)  # [N,1,128,96,96]
    new_landmark = torch.Tensor(np.vstack(new_landmark))  # [N,num,128,96,96]
    dataset = TensorDataset(new_image, new_mask, new_landmark)
    return dataset, location


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
        if mode =='max':
            result = torch.max(result, dim=0)[0]
        elif mode =='average':
            result = torch.mean(result,dim=0)
        else:
            raise ValueError(f"Unsupported mode:{mode}")
        return result.unsqueeze(0)


def generate_train_mask(landmark: torch.Tensor) -> torch.Tensor:
    '''landmark:[B,C,128,96,96], C is the number of landmark'''
    B = landmark.shape[0]
    shape = landmark.shape[2:]
    train_mask = torch.zeros_like(landmark)
    for b in range(B):
        indices = torch.where(landmark[b] > 0)[0]
        for idx in indices:
            train_mask[b, idx] = torch.ones(shape)
    return train_mask
