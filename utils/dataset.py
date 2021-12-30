import os
import torch
import random
import numpy as np
import SimpleITK as sitk
from typing import List, Tuple
from torch.utils.data import Dataset
from torch.utils.data.dataset import TensorDataset
from .util import generate_one_channel_landmark, random_sample, generate_landmark
from .augmentation import normalize, shift_scale_clamp, elastic_transformation


class SpineDataset(Dataset):
    def __init__(
        self,
        root: str,
        file_list: List[str],
        mode: str = 'train',
        landmark_num: int = 25,
    ) -> None:
        super().__init__()
        self.root = root
        self.landmark_num = landmark_num
        self.file_list = file_list
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.mode == 'train':
            return self._get_train_data(index)
        return self._get_inf_data(index)

    def _get_train_data(self, index):
        path = self.file_list[index]
        basename = os.path.basename(path)
        mask_path = os.path.join(self.root, basename)
        ID = basename[:basename.find('_seg.nii.gz')]
        img_path = mask_path.replace("_seg", "")
        image = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        image = normalize(image.astype(np.float32))
        # data augmentation
        # if np.random.uniform() > 0.5:
        #     image, mask = elastic_transformation(image, mask)
        # if np.random.uniform() > 0.5:
        #     image = shift_scale_clamp(image)
        # TODO
        # translation and rotation transform

        # crop patch
        image, mask = random_sample(image, mask)
        if self.landmark_num > 1:
            landmark = generate_landmark(mask, self.landmark_num)
        else:
            landmark = generate_one_channel_landmark(mask)
            landmark = np.expand_dims(landmark, 0)
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        return ID, image, mask, landmark

    def _get_inf_data(self, index):
        path = self.file_list[index]
        basename = os.path.basename(path)
        mask_path = os.path.join(self.root, basename)
        ID = basename[:basename.find('_seg.nii.gz')]
        img_path = mask_path.replace("_seg", "")
        image = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        image = normalize(image.astype(np.float32))
        if self.landmark_num > 1:
            landmark = generate_landmark(mask, self.landmark_num)
        else:
            landmark = generate_one_channel_landmark(mask)
            landmark = np.expand_dims(landmark, 0)
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        return ID, image, mask, landmark


class SegDataset(Dataset):
    def __init__(
        self,
        root: str,
        file_list: List[str],
        mode: str = 'train',
        patch_size: Tuple[int] = (128, 128, 96),
        augment: bool = False
    ) -> None:
        super().__init__()
        self.root = root
        self.file_list = file_list
        self.mode = mode
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.mode == 'train':
            return self._get_train_data(index)
        return self._get_inf_data(index)

    def _get_train_data(self, index):
        path = self.file_list[index]
        basename = os.path.basename(path)
        ID = basename[:basename.find("_seg.nii.gz")]
        mask_path = os.path.join(self.root, basename)
        img_path = mask_path.replace("_seg", "")
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        image = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        image, mask, landmark = self.generate_random_patch(image, mask)
        image = normalize(image.astype(np.float32))
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        landmark = np.expand_dims(landmark, 0)
        return ID, image, mask, landmark

    def _get_inf_data(self, index):
        path = self.file_list[index]
        basename = os.path.basename(path)
        ID = basename[:basename.find("_seg.nii.gz")]
        mask_path = os.path.join(self.root, basename)
        img_path = mask_path.replace("_seg", "")
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        image = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        classes = np.unique(mask)
        return ID, image, mask, classes[1:]

    def get_bbox(self, shape: np.ndarray, center: tuple):
        patch_size = list(self.patch_size)[::-1]
        # 原始图像的bbox | original bbox
        z_min = max(0, center[0] - patch_size[0]//2)
        z_max = min(shape[0], center[0] + patch_size[0]//2)
        y_min = max(0, center[1] - patch_size[1]//2)
        y_max = min(shape[1], center[1] + patch_size[1]//2)
        x_min = max(0, center[2] - patch_size[2]//2)
        x_max = min(shape[2], center[2] + patch_size[2]//2)

        # 新图像的bbox | new bbox
        Z_MIN = patch_size[0]//2 - (center[0] - z_min)
        Z_MAX = patch_size[0]//2 + (z_max - center[0])
        Y_MIN = patch_size[1]//2 - (center[1] - y_min)
        Y_MAX = patch_size[1]//2 + (y_max - center[1])
        X_MIN = patch_size[2]//2 - (center[2] - x_min)
        X_MAX = patch_size[2]//2 + (x_max - center[2])
        return (z_min, z_max, y_min, y_max, x_min, x_max), (Z_MIN, Z_MAX, Y_MIN, Y_MAX, X_MIN, X_MAX)

    def generate_random_patch(self, image, mask):
        classes = np.unique(mask)
        idx = random.randint(classes[1], classes[-1])
        Z, Y, X = np.where(mask == idx)
        Z = Z.mean().round().astype(int)
        Y = Y.mean().round().astype(int)
        X = X.mean().round().astype(int)
        if self.augment:
            Z += random.randint(-5, 5)
            Y += random.randint(-5, 5)
            X += random.randint(-5, 5)
        patch_size = list(self.patch_size)[::-1]
        bbox, BBOX = self.get_bbox(mask.shape, (Z, Y, X))

        new_mask = np.zeros(patch_size, dtype=np.uint8)
        new_img = -1023*np.ones(patch_size, dtype=np.int16)
        landmark = np.zeros(patch_size, dtype=np.float32)
        landmark[tuple([shape//2 for shape in patch_size])] = 1
        new_mask[
            BBOX[0]:BBOX[1],
            BBOX[2]:BBOX[3],
            BBOX[4]:BBOX[5]
        ] = mask[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
        new_img[
            BBOX[0]:BBOX[1],
            BBOX[2]:BBOX[3],
            BBOX[4]:BBOX[5]
        ] = image[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
        new_mask[new_mask != idx] = 0
        new_mask[new_mask > 0] = 1
        return new_img, new_mask, landmark

    def generate_inf_patch(self, image, mask):
        img_list = []
        mask_list = []
        patch_size = list(self.patch_size)[::-1]
        classes = np.unique(mask)[1:]
        for c in classes:
            new_mask = np.zeros(patch_size, dtype=np.uint8)
            new_img = -1023*np.ones(patch_size, dtype=np.int16)
            Z, Y, X = np.where(mask == c)
            Z = Z.mean().round().astype(int)
            Y = Y.mean().round().astype(int)
            X = X.mean().round().astype(int)
            bbox, BBOX = self.get_bbox(mask.shape, (Z, Y, X))
            new_mask[
                BBOX[0]:BBOX[1],
                BBOX[2]:BBOX[3],
                BBOX[4]:BBOX[5]
            ] = mask[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
            new_img[
                BBOX[0]:BBOX[1],
                BBOX[2]:BBOX[3],
                BBOX[4]:BBOX[5]
            ] = image[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
            new_mask[new_mask != c] = 0
            new_mask[new_mask > 0] = 1

            img_list.append(np.expand_dims(new_img, 0))
            mask_list.append(np.expand_dims(new_mask, 0))

        new_img = torch.Tensor(np.vstack(img_list)).unsqueeze(1)  # [N,1,96,128,128]
        new_img = torch.clip(new_img/2048, -1, 1)
        new_mask = torch.Tensor(np.vstack(mask_list)).unsqueeze(1)
        landmark = torch.zeros_like(new_mask, dtype=torch.float32)
        landmark[:, 0, patch_size[0]//2, patch_size[1]//2, patch_size[2]//2] = 1
        return TensorDataset(new_img, new_mask, landmark)
