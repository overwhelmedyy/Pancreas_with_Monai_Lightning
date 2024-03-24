import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
from monai.data import pad_list_data_collate, list_data_collate, decollate_batch, DataLoader, PersistentDataset, \
    Dataset, CacheDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet, SwinUNETR
from monai.transforms import (
    Compose, LoadImaged, RandCropByPosNegLabeld,
    RandAffined, RandRotated, RandFlipd, ResizeWithPadOrCropd, EnsureChannelFirstd, EnsureType, AsDiscrete
)
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter

args = argparse.Namespace(
    task_name="Task01_pancreas",
    network_name="SwinUNETR_pytorch",
    countiune_train=True,  # True ot False
    load_model_path=r"C:\Git\NeuralNetwork\Pancreas_with_Monai_Lightning\runs\Task01_pancreas\SwinUNETR_pytorch\0324_1911\checkpoint\model_147epochs.pth",
    no_cuda=False,  # disables CUDA training
    save_model_every_n_epoch=1,  # save model every epochs (default : 1)
    val_every_n_epoch=5,  # validate every epochs (default : 1)
    log_every_n_step=10,  # log every step (default : 1)
    epochs=1000,  # number of epochs (default : 10)
    lr=3e-4,  # base learning rate (default : 0.01)
    batch_size=2,  # batch size (default : 4)
    dry_run=False  # quickly check a single pass
)

time_stamp = datetime.now().strftime('%m%d_%H%M')


directory = os.environ.get("MONAI_DATA_DIRECTORY")
data_dir = os.path.join(directory, args.task_name)
persistent_cache = os.path.join(data_dir, "persistent_cache")

cuda = torch.device("cuda:0")


class TrainEval:

    def __init__(self, args, model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, metric, writer, device):
        self.train_loss_last_epoch = 0
        self.val_metric_latest = 0
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.metric = metric
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = args.epochs
        self.device = device
        self.writer = writer  # Added writter as a parameter
        self.args = args
        self.step = 0
        self.previous_model_path = None
        self.this_model_path = None
        self.highest_val_metric = -np.inf
        self.hvm_epoch = 0
        self.lowest_train_loss = np.inf
        self.ltl_epoch = 0
        self.scheduler = scheduler


        self.post_pred = Compose(
            [EnsureType("tensor", device=torch.device("cpu")), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device=torch.device("cpu")), AsDiscrete(to_onehot=2)])

    def train_fn(self, current_epoch):
        self.model.train()
        total_loss = []
        step_loss = []
        tk_step = tqdm(self.train_dataloader, desc="EPOCH"+"[TRAIN]"+str(current_epoch),
                       position=0, leave=True, dynamic_ncols=True)  # Added dynamic_ncols=True

        # 每循环一次应该是一个step
        for _, data in enumerate(tk_step):
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
            tk_step.set_postfix({f"train_loss_step":round(loss.item(), 5),
                                 f"train_loss_epoch":round(self.train_loss_last_epoch, 5),
                                 f"val_metric":round(self.val_metric_latest, 5),
                                 f"hvm{self.hvm_epoch}":round(self.highest_val_metric, 5),
                                 f"ltl{self.ltl_epoch}":round(self.lowest_train_loss,5)})

            # n个step记录loss
            if self.step % self.args.log_every_n_step == 0:
                step_ave_loss = round(sum(step_loss) / len(step_loss), 5)
                self.writer.add_scalar("train_loss_step", step_ave_loss, self.step)

        # 每个epoch记录平均loss
        ave_loss = round(sum(total_loss) / len(total_loss), 5)
        self.writer.add_scalar("train_loss_epoch", ave_loss, current_epoch)
        tk_step.set_postfix({"train_loss_epoch":round(ave_loss, 5)})

        # 保存模型
        # Save the new model
        if current_epoch % self.args.save_model_every_n_epoch == 0:
            self.this_model_path = os.path.join(os.path.join(self.writer.log_dir, "checkpoint"), f"model_{current_epoch}epochs.pth")
            torch.save(self.model.state_dict(), self.this_model_path)

            if self.previous_model_path is not None and os.path.exists(self.previous_model_path):
                os.remove(self.previous_model_path)
            # Update the path of the previous model
        self.previous_model_path = self.this_model_path

        return ave_loss

    def eval_fn(self, current_epoch):
        self.model.eval()
        tk_step = tqdm(self.val_dataloader, desc="EPOCH"+"[VALID]"+str(current_epoch), position=0, leave=True, dynamic_ncols=True)  # Added dynamic_ncols=True

        with torch.no_grad():
            for t, data in enumerate(tk_step):
                images, labels = data["image"], data["label"]
                images, labels = images.to(self.device), labels.to(self.device)

                logits = self.model(images)
                logits = [self.post_pred(i) for i in decollate_batch(logits)]
                labels = [self.post_label(i) for i in decollate_batch(labels)]
                self.metric(logits, labels)
                tk_step.set_postfix({"metric": "%4f" % round(self.metric.aggregate().item(), 5)})

            metric = self.metric.aggregate().item()
            tk_step.set_postfix({"metric": "%4f" % round(metric, 5)})
            self.writer.add_scalar("val_metric", metric, current_epoch)

        return metric

    def train(self):

        self.val_metric_latest = self.eval_fn(0)
        if self.val_metric_latest > self.highest_val_metric:
            self.highest_val_metric = self.val_metric_latest

        td = tqdm(range(self.epochs), desc="Training", position=0, leave=True, dynamic_ncols=True)  # Added dynamic_ncols=True
        # epochs循环
        for i in td:
            current_epoch = i + 1
            train_loss = self.train_fn(current_epoch)
            self.train_loss_last_epoch = train_loss

            if train_loss < self.lowest_train_loss:
                self.lowest_train_loss = train_loss
                self.ltl_epoch = current_epoch

            # 做validation
            if current_epoch % self.args.val_every_n_epoch == 0:
                self.val_metric_latest = self.eval_fn(current_epoch)
                print(f"val_metric{current_epoch}={round(self.val_metric_latest, 5)}")
                if self.val_metric_latest > self.highest_val_metric:
                    self.highest_val_metric = self.val_metric_latest
                    self.hvm_epoch = current_epoch
            self.scheduler.step()

def main():
    log_dir = fr"../runs/{args.task_name}/{args.network_name}/{time_stamp}"
    writter = SummaryWriter(log_dir)
    ckpt_dir = os.path.join(writter.log_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)


    torch.set_float32_matmul_precision("medium")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cuda:0")

    # 读数据的名字
    train_images = sorted(glob.glob(os.path.join(data_dir, "img_proc", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "pancreas_seg_proc", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in
                  zip(train_images, train_labels)]

    # 拆分成train和val
    train_files = data_dicts
    val_files = random.sample(data_dicts, round(0.2 * len(data_dicts)))

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64, 64, 64),
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
                spatial_size=(64, 64, 64),
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
            EnsureChannelFirstd(keys=["image", "label"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64, 64, 64),
                pos=3,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
                allow_smaller=True
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(64, 64, 64))
        ]
    )

    train_dataset = CacheDataset(data=train_files, transform=train_transforms)
    valid_dataset = CacheDataset(data=val_files, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=pad_list_data_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=pad_list_data_collate)

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
        img_size=(64, 64, 64)
    )

    if args.countiune_train:
        ckpt = torch.load(args.load_model_path)
        if "state_dict" in ckpt:
            new_state_dict = {}
            for key, value in ckpt["state_dict"].items():
                if key.startswith("_model."):
                    new_key = key[len("_model."):]  # Remove the "_module" prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            new_state_dict.pop("loss_function.class_weight")
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(ckpt)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150, 300], gamma=0.75)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    metric = DiceMetric(include_background=False, reduction="mean")

    TrainEval(args, model, train_loader, valid_loader, optimizer, scheduler, loss_function, metric, writter, device).train()


if __name__ == "__main__":
    main()
