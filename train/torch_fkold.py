import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import Compose, ToTensor, Resize
from torch import optim
from UXNet_3D.networks.UXNet_3D.network_backbone import UXNET

from torch.hub import tqdm
import glob
import os
import random

import lightning
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from monai.data import pad_list_data_collate, list_data_collate, decollate_batch, DataLoader, PersistentDataset, \
    Dataset, CacheDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet, SwinUNETR
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
from monai.utils import set_determinism

args = argparse.Namespace(
    task_name="Task01_pancreas",
    network_name="SwinUNETR_pytorch",
    countiune_train=True,  # True ot False
    load_model_path=r"",
    no_cuda=False,  # disables CUDA training
    save_model_every_n_epoch=1,  # save model every epochs (default : 1)
    val_every_n_epoch=5,  # validate every epochs (default : 1)
    log_every_n_step=10,  # log every step (default : 1)
    epochs=10,  # number of epochs (default : 10)
    lr=1e-4,  # base learning rate (default : 0.01)
    weight_decay=3e-3,  # weight decay value (default : 0.03)
    batch_size=2,  # batch size (default : 4)
    dry_run=False  # quickly check a single pass
)

directory = os.environ.get("MONAI_DATA_DIRECTORY")
data_dir = os.path.join(directory, args.task_name)
persistent_cache = os.path.join(data_dir, "persistent_cache")
tensorboard_dir = os.path.join("../runs", f"{args.task_name}")
tb_logger = TensorBoardLogger(tensorboard_dir, name=args.network_name)
ckpt_dir = os.path.join(tb_logger.log_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
cuda = torch.device("cuda:0")


class TrainEval:

    def __init__(self, args, model, train_dataloader, val_dataloader, optimizer, criterion, metric, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.metric = metric
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = args.epochs
        self.device = device
        self.args = args
        self.step = 0
        self.previous_model_path = None
        self.this_model_path = None

    def train_fn(self, current_epoch):
        self.model.train()
        total_loss = []
        step_loss = []
        tk = tqdm(self.train_dataloader, desc="EPOCH" + "[TRAIN]" + str(current_epoch + 1) + "/" + str(self.epochs))

        # 每循环一次应该是一个step
        for _, data in enumerate(tk):
            self.step += 1
            images, labels = data["image"], data["label"]
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss.append(loss.item())
            step_loss.append(loss.item())
            tk.set_postfix({"train_loss_step": "%4f" % float(round(loss.item(),3))})

            # n个step记录loss
            if self.step % self.args.log_every_n_step == 0:
                step_ave_loss = round(sum(step_loss) / len(step_loss), 4)
                tb_logger.experiment.add_scalar("train_loss_step", step_ave_loss, self.step)

            # 每个epoch记录平均loss
        ave_loss = round(sum(total_loss) / len(total_loss), 4)
        tb_logger.experiment.add_scalar(f"train_loss", ave_loss, current_epoch)
        tk.set_postfix({"train_loss_epoch": "%4f" % float(ave_loss)})

        # 保存模型
        # Save the new model
        if current_epoch % self.args.save_model_every_n_epoch == 0:
            self.this_model_path = os.path.join(ckpt_dir, f"model_{current_epoch}epochs.pth")
            torch.save(self.model.state_dict(), self.this_model_path)

            if self.previous_model_path is not None and os.path.exists(self.previous_model_path):
                os.remove(self.previous_model_path)
            # Update the path of the previous model
        self.previous_model_path = self.this_model_path

        return ave_loss

    def eval_fn(self, current_epoch):
        self.model.eval()
        total_metric = []
        tk = tqdm(self.val_dataloader, desc="EPOCH" + "[VALID]" + str(current_epoch + 1) + "/" + str(self.epochs))

        with torch.no_grad():
            for t, data in enumerate(tk):
                images, labels = data["image"], data["label"]
                images, labels = images.to(self.device), labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                total_metric += loss.item()
                metric = round(sum(total_metric) / len(total_metric), 4)
                tb_logger.experiment.add_scalar("val_metric", metric, current_epoch)
                tk.set_postfix({"metric": "%4f" % float(metric)})
                if self.args.dry_run:
                    break

        return metric

    def train(self):
        best_val_metric = -np.inf
        best_train_loss = np.inf

        # epochs循环
        for i in range(self.epochs):
            current_epoch = i + 1
            start_time = time.time()
            train_loss = self.train_fn(current_epoch)

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                print(f"lowest train Loss : {best_train_loss} at epochs={current_epoch}")

            # 做validation
            if current_epoch // self.args.val_every_n_epoch == 0:
                val_metric = self.eval_fn(current_epoch)

                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    print(f"highest val loss={best_val_metric} at epochs={current_epoch}")

            print(f"Epoch {current_epoch} took {round((time.time() - start_time) / 60, 2)} mins")


def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cuda:0")

    # model = UNet(
    #         spatial_dims=3,
    #         in_channels=1,
    #         out_channels=2,
    #         channels=(16, 32, 64, 128, 256),
    #         strides=(2, 2, 2, 2),
    #         num_res_units=2,
    #         norm=Norm.BATCH,
    #     ).to(device)

    # model = UXNET(
    #     in_chans=1,
    #     out_chans=2,
    #     depths=[2, 2, 2, 2],
    #     feat_size=[48, 96, 192, 384],
    #     drop_path_rate=0,
    #     layer_scale_init_value=1e-6,
    #     spatial_dims=3,
    # ).to(device)
    model = SwinUNETR(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        img_size=(96, 96, 96)
    ).to(device)

    if args.countiune_train:
        model.load_state_dict(args.load_model_path)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    metric = DiceMetric(include_background=False, reduction="mean")

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            # EnsureChannelFirstd(keys=["image", "label"]),
            # # LabelToMaskd(keys=['label'],select_labels=[0,2]),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Spacingd(
            #     keys=["image", "label"],
            #     pixdim=(1.5, 1.5, 2.0),
            #     mode=("bilinear", "nearest"),
            # ),
            # ScaleIntensityRanged(
            #     keys=["image"],
            #     a_min=-1024,
            #     a_max=2976,
            #     b_min=0.0,
            #     b_max=1.0,
            #     clip=True,
            # ),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            # # randomly crop out patch samples from
            # # big image based on pos / neg ratio
            # # the image centers of negative samples
            # # must be in valid image area
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
                prob=0.3,
                spatial_size=(96, 96, 96),
                rotate_range=(0, 0, np.pi / 15),
                scale_range=(0.1, 0.1, 0.1)),

            RandRotated(
                keys=['image', 'label'],
                range_x=np.pi / 4,
                range_y=np.pi / 4,
                range_z=np.pi / 4,
                prob=0.3,
                keep_size=True
            ),
            RandFlipd(
                keys=['image', 'label'],
                prob=0.3
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            # EnsureChannelFirstd(keys=["image", "label"]),
            # # LabelToMaskd(keys=['label'], select_labels=[0,2]),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Spacingd(
            #     keys=["image", "label"],
            #     pixdim=(1.5, 1.5, 2.0),
            #     mode=("bilinear", "nearest"),
            # ),
            # ScaleIntensityRanged(
            #     keys=["image"],
            #     a_min=-1024,
            #     a_max=2976,
            #     b_min=0.0,
            #     b_max=1.0,
            #     clip=True,
            # ),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
                allow_smaller=True
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(96, 96, 96))
        ]
    )

    # 读数据的名字
    images = sorted(glob.glob(os.path.join(data_dir, "img", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "pancreas_seg", "*.nii.gz")))

    kf = KFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(kf.split(images)):
        train_images, val_images = images[train_index], images[test_index]
        train_labels, val_labels = labels[train_index], labels[test_index]

    # data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in
    #               zip(train_images, train_labels)]
        train_files = [{"image": image_name, "label": label_name} for image_name, label_name in
                   zip(train_images, train_labels)]

        val_files = [{"image": image_name, "label": label_name} for image_name, label_name in
                 zip(val_images, val_labels)]

    # 拆分成train和val
    # train_files = random.sample(data_dicts, round(0.8 * len(data_dicts)))
    # val_files = [i for i in data_dicts if i not in train_files]

        train_dataset = CacheDataset(data=train_files, transform=train_transforms)
        valid_dataset = CacheDataset(data=val_files, transform=val_transforms)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  collate_fn=pad_list_data_collate)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  collate_fn=pad_list_data_collate)



        TrainEval(args, model, train_loader, valid_loader, optimizer, loss_function, metric, device).train()


if __name__ == "__main__":
    main()
