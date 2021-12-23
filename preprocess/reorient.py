'''
https://github.com/christianpayer/MedicalDataAugmentationTool-VerSe/blob/master/verse2019/other/reorient_reference_to_rai.py
'''
import os
import itk
import argparse
import numpy as np
from glob import glob


def reorient_to_rai(image):
    """
    Reorient image to RAI orientation.
    :param image: Input itk image.
    :return: Input image reoriented to RAI.
    """
    filter = itk.OrientImageFilter.New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    m = itk.GetMatrixFromArray(
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    reoriented = filter.GetOutput()
    return reoriented


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(os.path.join("data", "reoriented"), exist_ok=True)

    '''filename would be like: VerSe2019/train/verse007.nii.gz'''
    filenames = glob(os.path.join(args.dataset, '*', '*.nii.gz'))
    for filename in sorted(filenames):
        basename = os.path.basename(filename)
        basename_wo_ext = basename[:basename.find('.nii.gz')]
        is_seg = basename_wo_ext.endswith('_seg')
        print(basename_wo_ext)
        image = itk.imread(filename, itk.UC if is_seg else itk.SS)
        reoriented = reorient_to_rai(image)

        reoriented.SetOrigin([0, 0, 0])
        m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
        reoriented.SetDirection(m)
        reoriented.Update()
        itk.imwrite(reoriented, os.path.join('data', 'reoriented', basename_wo_ext + '.nii.gz'))
