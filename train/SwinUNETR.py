import glob
import os
import random
import numpy as np
import lightning
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from monai.data import list_data_collate, decollate_batch, DataLoader, PersistentDataset, CacheDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
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
network_name = "SwinUNETR"

directory = os.environ.get("MONAI_DATA_DIRECTORY")
data_dir = os.path.join(directory, task_name)
persistent_cache = os.path.join(data_dir, "persistent_cache")
tensorboard_dir = os.path.join("../runs", f"{task_name}")
cuda = torch.device("cuda:0")

set_determinism(seed=0)

tensorboard_logger = TensorBoardLogger(tensorboard_dir, name=network_name)

# 台式机显存跑满 batch_size = 8 134//8=17， log_every_n_steps=8,
train_batch_size = 2
val_batch_size = 2
learning_rate = 4e-4

class Net(lightning.LightningModule):
    def __init__(self, learning_rate, tr_bs, val_bs):
        super().__init__()
        self.val_ds = None
        self.train_ds = None
        self.learning_rate = learning_rate
        self.tr_bs = tr_bs
        self.val_bs = val_bs
        self._model = SwinUNETR(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            img_size=(96, 96, 96)
        )

        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose(
            [EnsureType("tensor", device=torch.device("cpu")), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device=torch.device("cpu")), AsDiscrete(to_onehot=2)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []
        self.tb_logger = tensorboard_logger.experiment

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        # set up the correct data path
        train_images = sorted(glob.glob(os.path.join(data_dir, "img_proc", "*.nii.gz")))
        train_labels = sorted(glob.glob(os.path.join(data_dir, "pancreas_seg_proc", "*.nii.gz")))
        data_dicts = [
            {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)
        ]  # 注意到data_dicts是一个数组
        # train_files, val_files = random_split(data_dicts, [0.8, 0.2])
        train_files = random.sample(data_dicts, round(0.85 * len(data_dicts)))
        val_files = [i for i in data_dicts if i not in train_files]

        # set deterministic training for reproducibility
        set_determinism(seed=0)

        # define the data transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                # LabelToMaskd(keys=['label'],select_labels=[0,2]),
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
                # randomly crop out patch samples from
                # big image based on pos / neg ratio
                # the image centers of negative samples
                # must be in valid image area
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(64, 64, 64),
                    pos=6,
                    neg=1,
                    num_samples=4,
                    allow_smaller=True,
                    image_key="image",
                    # image_threshold=0,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=0.3,
                    spatial_size=(64, 64, 64),
                    rotate_range=(0, 0, np.pi/15),
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
                # Rand3DElasticd(
                #     keys=['image', 'label'],
                #     sigma_range=(5, 8),
                #     magnitude_range=(100, 200),
                #     prob=0.5,
                #     rotate_range=(0, 0, np.pi / 15),
                #     scale_range=(0.2, 0.2, 0.2),
                #     spatial_size=(96, 96, 96),
                #     mode=('bilinear', 'nearest')),

                ResizeWithPadOrCropd(keys=["image", "label"],
                                     spatial_size=(64, 64, 64),
                                     mode='constant')
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
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
                    spatial_size=(64, 64, 64),
                    pos=6,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True
                ),
                ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(64, 64, 64))
            ]
        )

        # we use cached datasets - these are 10x faster than regular datasets
        self.train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
        )
        self.val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
        )


        # self.train_ds = monai.data.Dataset(
        #     data=train_files, transform=train_transforms)
        # self.val_ds = Dataset(
        #     data=val_files, transform=val_transforms)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.tr_bs,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            collate_fn=list_data_collate,  # 把每个batch的list of data合并到一个list中
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.val_bs,
            num_workers=4,
            persistent_workers=True
        )
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 300], gamma=0.5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        self.log("train_loss", loss.item(), prog_bar=True, logger=True, on_epoch=True)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.validation_step_outputs.clear()
        self.log_dict(tensorboard_logs, logger=True)
        return {"log": tensorboard_logs}

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    net = Net(learning_rate=learning_rate,
              tr_bs=train_batch_size,
              val_bs=val_batch_size
              )

    trainer = lightning.Trainer(
        devices="auto",
        max_epochs=1000,
        logger=tensorboard_logger,
        log_every_n_steps=10,
        enable_checkpointing=True,
        deterministic=True,
        check_val_every_n_epoch=5,
        num_sanity_val_steps=None
    )

    module_resume = Net.load_from_checkpoint(r"C:\Git\NeuralNetwork\Pancreas_with_Monai_Lightning\runs\Task01_pancreas\SwinUNETR\version_12\checkpoints\epoch=59-step=3780.ckpt",
                                             learning_rate=learning_rate,
                                             tr_bs=train_batch_size,
                                             val_bs=val_batch_size)

    trainer.fit(module_resume)
    print(f"train completed, best_metric: {net.best_val_dice:.4f} " f"at epoch {net.best_val_epoch}")
#