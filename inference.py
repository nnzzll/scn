import os
import glob
import time
import torch
import SimpleITK as sitk
from utils.infer import Image
from network import LocalAppearance, UNet


if __name__ == '__main__':
    os.makedirs('result/seg_result', exist_ok=True)
    scn = LocalAppearance(1, 1).cuda()
    scn.eval()
    scn.load_state_dict(torch.load("run/SCN_one_channel/model.pth")["weight"])

    unet = UNet(2, 1).cuda()
    # unet.eval()
    unet.load_state_dict(torch.load("run/unet/best_checkpoint.pth")['model_state_dict'])

    files = sorted(glob.glob("data/reoriented/*_seg.nii.gz"))
    for f in files:
        basename = os.path.basename(f)
        ID = basename[:basename.find("_seg.nii.gz")]
        mask_path = f"data/reoriented/{basename}"
        img_path = mask_path.replace("_seg", "")
        patient = Image(img_path, 0, mask_path)

        begin = time.time()
        landmark = patient.landmark_localization(scn)
        new_landmark = patient.landmark_transform(landmark)
        seg = patient.vertebrae_segmentation(unet,new_landmark)
        end = time.time()
        seg_img = sitk.GetImageFromArray(seg)
        seg_img.SetOrigin(patient.standard_img.GetOrigin())
        sitk.WriteImage(seg_img, f"result/end2end_result/{ID}_pred.nii.gz")
        print(f"ID:{ID}\tTime:{end-begin:.3f}s.")
