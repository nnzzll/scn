import os
import glob
import argparse
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool


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


def crop(path: str):
    newSize = [96, 96, 128]
    newSpacing = (2, 2, 2)
    basename = os.path.basename(path)
    basename_wo_ext = basename[:basename.find("_seg.nii.gz")]
    img = sitk.ReadImage(path)
    resampled = resample(img, sitk.sitkNearestNeighbor, newSpacing)
    arr: np.ndarray = sitk.GetArrayFromImage(resampled)
    size = resampled.GetSize()
    origin = resampled.GetOrigin()

    if size[2] >= newSize[2]:
        newSize[2] = size[2]
    Z, Y, X = np.where(arr > 0)
    center = ((X.max()+X.min())//2, (Y.max()+Y.min())//2, (Z.max()+Z.min())//2)

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

    new_arr = np.zeros(tuple(newSize[::-1]), dtype=np.uint8)
    new_arr[
        Z_MIN:Z_MAX,
        Y_MIN:Y_MAX,
        X_MIN:X_MAX
    ] = arr[z_min:z_max, y_min:y_max, x_min:x_max]
    new_img = sitk.GetImageFromArray(new_arr)
    new_img.SetOrigin(newOrigin)
    new_img.SetSpacing(newSpacing)
    sitk.WriteImage(new_img, f"data/2mm/{basename}")

    img = sitk.ReadImage(f'data/reoriented/{basename_wo_ext}.nii.gz')
    resampled = resample(img, sitk.sitkLinear, newSpacing)
    arr = sitk.GetArrayFromImage(resampled)
    new_arr = -1023*np.ones(tuple(newSize[::-1]), dtype=np.int16)
    new_arr[
        Z_MIN:Z_MAX,
        Y_MIN:Y_MAX,
        X_MIN:X_MAX
    ] = arr[z_min:z_max, y_min:y_max, x_min:x_max]
    new_img = sitk.GetImageFromArray(new_arr)
    new_img.SetOrigin(newOrigin)
    new_img.SetSpacing(newSpacing)
    sitk.WriteImage(new_img, f"data/2mm/{basename_wo_ext}.nii.gz")
    print(basename, "Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--process", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(os.path.join("data", "2mm"), exist_ok=True)
    files = sorted(glob.glob("data/reoriented/*_seg.nii.gz"))

    mp = Pool(args.process)
    mp.map(crop, files)
