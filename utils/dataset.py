import os
import numpy as np
import SimpleITK as sitk
from typing import Dict, List
from torch.utils.data import Dataset
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
