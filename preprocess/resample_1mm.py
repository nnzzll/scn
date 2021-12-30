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


def transform(path: str):
    basename = os.path.basename(path)
    basename_wo_ext = basename[:basename.find("_seg.nii.gz")]
    img = sitk.ReadImage(path)
    resampled = resample(img, sitk.sitkNearestNeighbor)
    sitk.WriteImage(resampled, f"data/1mm/{basename}")

    img = sitk.ReadImage(f'data/reoriented/{basename_wo_ext}.nii.gz')
    resampled = resample(img, sitk.sitkLinear)
    sitk.WriteImage(resampled, f'data/1mm/{basename_wo_ext}.nii.gz')
    print(basename_wo_ext, "Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--process", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(os.path.join("data", "1mm"), exist_ok=True)
    files = sorted(glob.glob("data/reoriented/*_seg.nii.gz"))

    mp = Pool(args.process)
    mp.map(transform, files)
