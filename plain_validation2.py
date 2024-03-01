import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import Compose, ToTensor, Resize
from torch import optim

from torch.hub import tqdm
import glob
import os
import random

import lightning
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from monai.data import pad_list_data_collate, list_data_collate, decollate_batch, DataLoader, PersistentDataset,Dataset, CacheDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
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
    EnsureType, RandAffined, RandRotated, RandFlipd, Rand3DElasticd, ResizeWithPadOrCropd
)


task_name = "Task01_pancreas"
network_name = "UNet"

directory = os.environ.get("MONAI_DATA_DIRECTORY")
data_dir = os.path.join(directory, task_name)
persistent_cache = os.path.join(data_dir, "persistent_cache")
tensorboard_dir = os.path.join("./runs", f"{task_name}")
cuda = torch.device("cuda:0")

args = argparse.Namespace(
    no_cuda=False,                  # disables CUDA training
    # patch_size=16,                  # patch size for images (default : 16)
    # latent_size=768,                # latent size (default : 768)
    # n_channels=3,                   # number of channels in images (default : 3 for RGB)
    # num_heads=12,                   # (default : 12)
    # num_encoders=12,                # number of encoders (default : 12)
    # dropout=0.1,                    # dropout value (default : 0.1)
    # img_size=224,                   # image size to be reshaped to (default : 224)
    # num_classes=10,                 # number of classes in dataset (default : 10 for CIFAR10)
    epochs=10,                      # number of epochs (default : 10)
    lr=1e-2,                        # base learning rate (default : 0.01)
    weight_decay=3e-2,              # weight decay value (default : 0.03)
    batch_size=1,                   # batch size (default : 4)
    dry_run=False                  # quickly check a single pass
)


class PlainVal:

    def __init__(self, args, model, val_dataloader, criterion, device):
        self.model = model

        self.val_dataloader = val_dataloader

        self.criterion = criterion
        self.epoch = args.epochs
        self.device = device
        self.args = args

    def eval_fn(self, current_epoch):
        self.model.eval()
        total_loss = 0.0
        tk = tqdm(self.val_dataloader, desc="EPOCH" + "[VALID]" + str(current_epoch + 1) + "/" + str(self.epoch))

        for t, data in enumerate(tk):
            images, labels = data["image"], data["label"]
            images, labels = images.to(self.device), labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})
            if self.args.dry_run:
                break

        return total_loss / len(self.val_dataloader)

    def valdation(self):
        best_valid_loss = np.inf
        for i in range(self.epoch):
            val_loss = self.eval_fn(i)

            if val_loss < best_valid_loss:
                print("Saved Best Weights")
                best_valid_loss = val_loss

            print(f"valid Loss : {val_loss}")
        print(f"Best valid Loss : {best_valid_loss}")

    '''
        On default settings:

        Training Loss : 2.3081023390197752
        Valid Loss : 2.302861615943909

        However, this score is not competitive compared to the 
        high results in the original paper, which were achieved 
        through pre-training on JFT-300M dataset, then fine-tuning 
        it on the target dataset. To improve the model quality 
        without pre-training, we could try training for more epochs, 
        using more Transformer layers, resizing images or changing 
        patch size,
    '''


def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 读数据的名字
    train_images = sorted(glob.glob(os.path.join(data_dir, "img", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "pancreas_seg", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]  # 注意到data_dicts是一个数组

    # 拆分成train和val
    validation_files = random.sample(data_dicts, round(0.3 * len(data_dicts)))


    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # LabelToMaskd(keys=['label'],select_labels=[0,2]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1024,
                a_max=2976,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # randomly crop out patch samples from
            # big image based on pos / neg ratio
            # the image centers of negative samples
            # must be in valid image area
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),

            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0,
                spatial_size=(96, 96, 96),
                rotate_range=(0, 0, np.pi / 15),
                scale_range=(0.1, 0.1, 0.1)),

            RandRotated(
                keys=['image', 'label'],
                range_x=np.pi / 4,
                range_y=np.pi / 4,
                range_z=np.pi / 4,
                prob=0.4,
                keep_size=True
            ),
            RandFlipd(
                keys=['image', 'label'],
                prob=0.5
            ),
            Rand3DElasticd(
                keys=['image', 'label'],
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                prob=0.5,
                rotate_range=(0, 0, np.pi / 15),
                scale_range=(0.2, 0.2, 0.2),
                spatial_size=(96, 96, 96),
                mode=('bilinear', 'nearest'))
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # LabelToMaskd(keys=['label'], select_labels=[0,2]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1024,
                a_max=2976,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(160, 160, 160),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
                allow_smaller=True
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(160, 160, 160))
        ]
    )

    validation_dataset = PersistentDataset(data=validation_files, transform=val_transforms,cache_dir=persistent_cache)
    valid_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=pad_list_data_collate, num_workers=4)

    model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
    ckpt = torch.load("runs/Task01_pancreas/UNet/version_36/checkpoints/epoch=99-step=6700.ckpt")

    new_state_dict = {}
    for key, value in ckpt["state_dict"].items():
        if key.startswith("_model."):
            new_key = key[len("_model."):]  # Remove the "_module" prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    new_state_dict.pop("loss_function.class_weight")
    model.load_state_dict(new_state_dict)
    model.to(device)
    criterion = DiceLoss(to_onehot_y=True, softmax=True)

    PlainVal(args, model, valid_loader, criterion, device).valdation()


if __name__ == "__main__":
    main()
