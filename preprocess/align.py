import glob
import SimpleITK as sitk
import os

import numpy as np
import torch
from tqdm import tqdm
from scipy import ndimage
import glob
import os
from matplotlib import pyplot as plt
import torch

from monai.transforms import (
    Compose,
    ScaleIntensityRange,
)

task_name = "Task82_pancreas"
network_name = "UNet"
directory = os.environ.get("MONAI_DATA_DIRECTORY")
data_dir = os.path.join(directory, task_name)
persistent_cache = os.path.join(data_dir, "persistent_cache")
tensorboard_dir = os.path.join("../runs", f"{task_name}")
cuda = torch.device("cuda:0")


expand_slice = 48

preprocess_transforms_img = Compose(
    [
        ScaleIntensityRange(a_min=-150, a_max=150, b_min=0.0, b_max=1.0, clip=True),
    ]
)

train_images = sorted(glob.glob(os.path.join(data_dir, "img", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "pancreas_seg", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)
]  # 注意到data_dicts是一个数组

origin = (0, 0, 0)
spacing = (2.6, 2.6, 3)
direction =(1, 0, 0, 0, 1, 0, 0, 0, 1)

for _, i in tqdm(enumerate(data_dicts)):
    image = sitk.ReadImage(i["image"])
    label = sitk.ReadImage(i["label"])

    image_array = sitk.GetArrayFromImage(image)[:, 50:450, 50:450]
    label_array = sitk.GetArrayFromImage(label)[:, 50:450, 50:450]

    img_zoomed = ndimage.zoom(image_array, (0.7, 0.7, 0.7), order=3)
    label_zoomed = ndimage.zoom(label_array, (0.7, 0.7, 0.7), order=0)

    img_proc = preprocess_transforms_img(img_zoomed)
    label_proc = label_zoomed

    # 找到肝脏区域开始和结束的slice，并各向外扩张
    z = np.any(label_proc, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]

    # 两个方向上各扩张个slice
    start_slice = start_slice - expand_slice if start_slice - expand_slice > 0 else 0
    end_slice = end_slice + expand_slice if end_slice + expand_slice <= img_proc.shape[0] else img_proc.shape[0]-1

    img_proc_z = img_proc[start_slice:end_slice + 1, :, :]
    label_proc_z = label_proc[start_slice:end_slice + 1, :, :]

    img_output = sitk.GetImageFromArray(img_proc)
    # img_output.SetOrigin(origin)
    # img_output.SetSpacing(spacing)
    # img_output.SetDirection(direction)

    label_output = sitk.GetImageFromArray(label_proc)
    # label_output.SetOrigin(origin)
    # label_output.SetSpacing(spacing)
    # label_output.SetDirection(direction)

    if img_output.GetDirection != label_output.GetDirection:
        print(f"{_} Direction is not the same")
        print(img_output.GetDirection())
        print(label_output.GetDirection())
    if img_output.GetSpacing != label_output.GetSpacing:
        print(f"{_} Spacing is not the same")
        print(img_output.GetSpacing())
        print(label_output.GetSpacing())
    if img_output.GetOrigin != label_output.GetOrigin:
        print(f"{_} Origin is not the same")
        print(img_output.GetOrigin())
        print(label_output.GetOrigin())

    sitk.WriteImage(img_output, os.path.join(data_dir, "img_proc",f"img_{_:04d}.nii.gz"))
    sitk.WriteImage(label_output, os.path.join(data_dir, "pancreas_seg_proc",f"pancreas_seg_{_:04d}.nii.gz"))










