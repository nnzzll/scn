import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List
from scipy.ndimage import gaussian_filter
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from utils.filters import Gaussian

from utils.util import generate_one_channel_landmark

from .postprocess import local_maxima, process_coords, one_class_dist
from .transform import resample, Aggregator, normalize


class Image(object):
    def __init__(
        self,
        path: str,
        num_classes: int = 0,
        mask: str = None,
        device: str = 'cuda'
    ) -> None:
        super().__init__()
        self.origin_img = sitk.ReadImage(path)
        self.standard_img = resample(self.origin_img)
        self.num_classes = num_classes
        self.device = device
        if mask:
            mask = sitk.ReadImage(mask)
            self.landmark_img = crop_to_2mm(self.origin_img, mask)
            self.standard_mask = resample(mask, sitk.sitkNearestNeighbor)

    def spine_localization(self, sigma: float = 3.0):
        # TODO
        raise NotImplementedError

    def landmark_localization(self, model):
        num = self.num_classes if self.num_classes else 1
        agg = Aggregator(num)
        img = sitk.GetArrayFromImage(self.landmark_img)
        img = normalize(img)
        patches, location = generate_patches(img)
        with torch.no_grad():
            for i, (img,) in enumerate(patches):
                if self.num_classes:
                    pred, _, _ = model(img.to(self.device))
                else:
                    _, pred = model(img.to(self.device))
                agg.Add(pred, location[i])
            pred = agg.Execute()
        self.landmark_heatmap = pred.cpu().squeeze().numpy()
        pred = np.clip(self.landmark_heatmap, 0, 1)
        torch.cuda.empty_cache()
        return get_one_class_landmark(pred)

    def vertebrae_segmentation(self, model, landmark) -> np.ndarray:
        shape = list(self.standard_img.GetSize())[::-1]
        patches = crop_patch(self.standard_img, landmark)
        heatmap = generate_heatmap(5, 2)
        heatmap = torch.Tensor(heatmap)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        result = np.zeros((len(patches)+1, *shape), dtype=np.float32)
        for i, (img,new_landmark) in enumerate(patches):
            inputs = torch.cat([img, heatmap], dim=1)
            with torch.no_grad():
                output = model(inputs.to(self.device))
            output = output.squeeze().cpu().numpy()
            output[output < 0.5] = 0
            bbox, BBOX = get_bbox(shape, landmark[i])
            result[0, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = 1
            result[
                i+1,
                bbox[0]:bbox[1],
                bbox[2]:bbox[3],
                bbox[4]:bbox[5]
            ] = output[BBOX[0]:BBOX[1], BBOX[2]:BBOX[3], BBOX[4]:BBOX[5]]
        torch.cuda.empty_cache()
        result[0] = 1 - result[0]
        result = np.argmax(result, 0).astype(np.uint8)
        return result

    def landmark_transform(self, landmark: np.ndarray):
        '''transform landmark coordination from 2mm spacing to 1mm spacing'''
        new_landmark = np.zeros_like(landmark)
        origin = self.landmark_img.GetOrigin()
        spacing = self.landmark_img.GetSpacing()
        newOrigin = self.standard_img.GetOrigin()
        newSpacing = self.standard_img.GetSpacing()
        # landmark:[N,3], Z,Y,X
        # origin,spacing: X,Y,Z
        for i in range(len(landmark)):
            new_landmark[i, 0] = (
                origin[2]-newOrigin[2] + spacing[2]*landmark[i, 0])/newSpacing[2]
            new_landmark[i, 1] = (
                origin[1]-newOrigin[1] + spacing[1]*landmark[i, 1])/newSpacing[1]
            new_landmark[i, 2] = (
                origin[0]-newOrigin[0] + spacing[0]*landmark[i, 2])/newSpacing[0]

        return new_landmark

    def test_landmark(self, landmark):
        # TODO
        mask = sitk.GetArrayFromImage(self.standard_mask)
        gt_landmark = generate_one_channel_landmark(mask)
        Z, Y, X = np.where(gt_landmark)
        gt_landmark = np.array([Z, Y, X]).T
        return one_class_dist(gt_landmark, landmark)

    def test_segmentation(self, seg, landmark, threshold=12.5) -> Dict[int,float]:
        gt = sitk.GetArrayFromImage(self.standard_mask)
        gt_landmark = generate_one_channel_landmark(gt)
        gt_landmark = np.array([*np.where(gt_landmark)]).T
        classes = np.unique(gt)[1:].tolist()[::-1]
        pred_classes = np.unique(seg)[1:].tolist()
        dice = {}
        i = 0  # index of grountruth
        j = 0  # index of prediction
        while(i < len(gt_landmark) and j < len(landmark)):
            pt_gt = gt_landmark[i]
            pt_pr = landmark[j]
            if dist(pt_gt, pt_pr) < threshold:
                pred = seg == pred_classes[j]
                mask = gt == classes[i]
                dice[classes[i]] = Dice(pred, mask)
                i += 1
                j += 1
                continue
            if landmark[j, 0] < landmark[i, 0]:
                j += 1
            else:
                i += 1
        return dice


def get_bbox(shape, center, patch_size: List[int] = [96, 128, 128]):
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


def generate_heatmap(sigma: float = 3.0, trunc: int = 2, patch_size: List[int] = [96, 128, 128]):
    heatmap = np.zeros(patch_size, dtype=np.float32)
    heatmap[48, 64, 64] = 1
    heatmap = gaussian_filter(
        heatmap, sigma, mode='constant', truncate=trunc)
    return heatmap/heatmap.max()


def get_one_class_landmark(img: np.ndarray, val_th: float = 0.3, dist_th: float = 12.5):
    landmark = local_maxima(img, val_th)
    landmark = process_coords(landmark, dist_th)
    return landmark


def generate_patches(img: np.ndarray, overlap: int = 96) -> Tuple[DataLoader, List[int]]:
    shape = img.shape
    dz = 128 - overlap
    z_range = list(range(0, shape[0]-128, dz))
    z_range.append(shape[0]-128)
    z_range = sorted(list(set(z_range)))
    location = []
    new_img = []
    for z in z_range:
        new_img.append(np.expand_dims(img[z:z+128], 0))
        location.append(z)
    new_img = torch.Tensor(np.vstack(new_img)).unsqueeze(1)
    dataset = TensorDataset(new_img)
    patches = DataLoader(dataset, 1)
    return patches, location


def crop_to_2mm(img: sitk.Image, mask: sitk.Image) -> sitk.Image:
    newSize = [96, 96, 128]
    newSpacing = (2, 2, 2)

    resampled_mask = resample(mask, sitk.sitkNearestNeighbor, newSpacing)
    arr: np.ndarray = sitk.GetArrayFromImage(resampled_mask)
    size = resampled_mask.GetSize()
    origin = resampled_mask.GetOrigin()
    if size[2] >= newSize[2]:
        newSize[2] = size[2]
    Z, Y, X = np.where(arr > 0)
    center = ((X.max()+X.min())//2, (Y.max()+Y.min()) //
              2, (Z.max()+Z.min())//2)

    # 原始图像的bbox | original bbox
    x_min = max(0, center[0]-newSize[0]//2)
    x_max = min(size[0], center[0]+newSize[0]//2)
    y_min = max(0, center[1]-newSize[1]//2)
    y_max = min(size[1], center[1]+newSize[1]//2)
    z_min = max(0, center[2]-newSize[2]//2) if size[2] != newSize[2] else 0
    z_max = min(size[2], center[2]+newSize[2] //
                2) if size[2] != newSize[2] else size[2]

    # 新图像的bbox | new bbox
    X_MIN = newSize[0]//2-(center[0]-x_min)
    X_MAX = newSize[0]//2+(x_max-center[0])
    Y_MIN = newSize[1]//2-(center[1]-y_min)
    Y_MAX = newSize[1]//2+(y_max-center[1])
    Z_MIN = newSize[2]//2-(center[2]-z_min) if size[2] != newSize[2] else 0
    Z_MAX = newSize[2]//2 + \
        (z_max-center[2]) if size[2] != newSize[2] else size[2]

    # 新Origin | new origin
    newOrigin = [0, 0, 0]
    newOrigin[0] = origin[0] + (center[0] - newSize[0]//2)*newSpacing[0]
    newOrigin[1] = origin[1] + (center[1] - newSize[1]//2)*newSpacing[1]
    newOrigin[2] = origin[2] + (center[2] - newSize[2]//2) * \
        newSpacing[2] if size[2] != newSize[2] else origin[2]

    resampled_img = resample(img, sitk.sitkLinear, newSpacing)
    arr = sitk.GetArrayFromImage(resampled_img)
    new_arr = -1023*np.ones(tuple(newSize[::-1]), dtype=np.int16)
    new_arr[
        Z_MIN:Z_MAX,
        Y_MIN:Y_MAX,
        X_MIN:X_MAX
    ] = arr[z_min:z_max, y_min:y_max, x_min:x_max]
    new_img = sitk.GetImageFromArray(new_arr)
    new_img.SetOrigin(newOrigin)
    new_img.SetSpacing(newSpacing)
    return new_img


def crop_patch(img: sitk.Image, landmark: np.ndarray, mask: sitk.Image = None, patch_size: List[int] = [96, 128, 128]):
    new_img = []
    new_mask = []
    img = sitk.GetArrayFromImage(img)
    mask = sitk.GetArrayFromImage(mask) if mask else None
    classes = np.unique(mask)[1:].tolist()[::-1]
    for i in range(len(landmark)):
        tmp = -1023*np.ones(patch_size, dtype=np.int16)
        bbox, BBOX = get_bbox(img.shape, landmark[i], patch_size)
        tmp[
            BBOX[0]:BBOX[1],
            BBOX[2]:BBOX[3],
            BBOX[4]:BBOX[5]
        ] = img[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
        new_img.append(np.expand_dims(tmp, 0))
        if mask is not None:
            tmp = np.zeros(patch_size, dtype=np.uint8)
            tmp[
                BBOX[0]:BBOX[1],
                BBOX[2]:BBOX[3],
                BBOX[4]:BBOX[5]
            ] = mask[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
            c = classes[i]
            tmp[tmp!=c] = 0
            tmp[tmp==c] = 1
            new_mask.append(np.expand_dims(tmp, 0))
    new_img = torch.Tensor(np.vstack(new_img)).unsqueeze(1)
    if mask is not None:
        new_mask = torch.Tensor(np.vstack(new_mask)).unsqueeze(1)
    new_img = torch.clip(new_img/2048, -1, 1)
    new_landmark = torch.zeros_like(new_img,dtype=torch.float32)
    new_landmark[:, 0, patch_size[0]//2, patch_size[1]//2, patch_size[2]//2] = 1

    dataset = TensorDataset(new_img, new_mask,new_landmark) if mask is not None else TensorDataset(new_img,new_landmark)
    patches = DataLoader(dataset, 1)
    return patches


def Dice(pred: np.ndarray, target: np.ndarray, smooth: int = 1) -> float:
    pred = np.atleast_1d(pred.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    intersction = np.count_nonzero(pred & target)

    dice_coef = (2.*intersction+smooth)/float(
        np.count_nonzero(pred)+np.count_nonzero(target)+smooth
    )
    return dice_coef


def dist(x, y):
    return np.sqrt(np.power(x-y, 2).sum())
