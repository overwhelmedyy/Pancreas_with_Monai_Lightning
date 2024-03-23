import numpy as np
import glob
import os
import random

import lightning
from matplotlib import pyplot as plt

import monai
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from monai.data import pad_list_data_collate, list_data_collate, decollate_batch, DataLoader, PersistentDataset,Dataset, CacheDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet,ViT
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

li = LoadImaged(keys=["image", "label"])
data_dicts[0] = li(data_dicts[0])
image, label = data_dicts[0]["image"], data_dicts[0]["label"]
plt.figure("visualise", (8, 4))
plt.subplot(1, 2, 1)
plt.title("image_")
plt.imshow(image[0, :, :, 50], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label_")
plt.imshow(label[0, :, :, 50])
plt.show()


preprocess_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)
data_dicts[0] = preprocess_transforms(data_dicts[0])
image, label = data_dicts[0]["image"], data_dicts[0]["label"]
plt.figure("visualise", (8, 4))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[0, :, :, 50], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[0, :, :, 50])
plt.show()







# tensorboard_logger = TensorBoardLogger(tensorboard_dir, name=network_name)
# def cut_black_edges(image_data, num_voxels=20):
#     # Calculate the shape of the image
#     shape = image_data.shape
#
#     # Define the slicing ranges for each dimension
#     x_slice = slice(num_voxels, shape[0] - num_voxels)
#     y_slice = slice(num_voxels, shape[1] - num_voxels)
#     z_slice = slice(num_voxels, shape[2] - num_voxels)
#
#     # Apply the slicing to remove the black edges
#     cropped_image = image_data[x_slice, y_slice, z_slice]
#
#     return cropped_image
#
# # Example usage
# # Assuming 'preprocessed_image' contains your preprocessed image data
# # preprocessed_image = preprocess_image(image_path, zoom_factor)
# cropped_image = cut_black_edges(preprocessed_image)

# Now 'cropped_image' contains the preprocessed image with black edges removed
