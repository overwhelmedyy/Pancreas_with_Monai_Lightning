import glob
import os
from matplotlib import pyplot as plt
import torch

from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureType,
    RandAffined,
    RandRotated,
    RandFlipd,
    Rand3DElasticd,
    ResizeWithPadOrCropd
)
from monai.utils import set_determinism

task_name = "Task01_pancreas"
network_name = "UNet"

directory = os.environ.get("MONAI_DATA_DIRECTORY")
data_dir = os.path.join(directory, task_name)
persistent_cache = os.path.join(data_dir, "persistent_cache")
tensorboard_dir = os.path.join("../runs", f"{task_name}")
cuda = torch.device("cuda:0")


train_images = sorted(glob.glob(os.path.join(data_dir, "img", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "pancreas_seg", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)
]  # 注意到data_dicts是一个数组



for i in data_dicts:
    i = li(i)
    image = i["image"]
    label = i["label"]
    




# preprocess_transforms = Compose(
#     [
#         # LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         ScaleIntensityRanged(
#             keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
#         ),
#         CropForegroundd(keys=["image", "label"], source_key="image"),
#     ]
# )
