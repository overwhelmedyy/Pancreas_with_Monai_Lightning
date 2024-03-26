import argparse
import glob
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.hub import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

from monai.data import pad_list_data_collate, decollate_batch, DataLoader, CacheDataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose, LoadImaged, RandCropByPosNegLabeld,
    RandAffined, RandRotated, RandFlipd, ResizeWithPadOrCropd, EnsureChannelFirstd, EnsureType, AsDiscrete
)

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

    def __init__(self, args, model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, metric, writer,
                 device):
        # 训练过程中的指标
        self.train_loss_last_epoch = 0  # 上个epoch的loss
        self.val_metric_latest = 0  # 上一次val的metric
        self.hvm_epoch = 0  # 取得highest val metric对应的epoch
        self.ltl_epoch = 0  # 取得lowest train loss对应的epoch
        self.step = 0  # 训练一个batch算一个step
        self.highest_val_metric = -np.inf  # 历次val的最高metric
        self.lowest_train_loss = np.inf  # 所有epoch中的最低loss

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.criterion = criterion  # 训练用的loss（DiceLoss）
        self.metric = metric  # val用的metric指标（DSC）
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.epochs = args.epochs  # 一共要训练多少个epoch
        self.device = device
        self.writer = writer  # tensorboard summarywriter
        self.args = args  # 超参数包
        self.previous_model_path = None  # 保存上一个模型的路径
        self.this_model_path = None  # 保存此次模型的路径 先存新的，再删旧的

        # 我觉得需要后处理，但是没有好像也能跑
        self.post_pred = Compose(
            [EnsureType("tensor", device=torch.device("cpu")), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device=torch.device("cpu")), AsDiscrete(to_onehot=2)])

    def train_fn(self, current_epoch):
        self.model.train()
        total_loss = []  # 分别计录epoch loss和step loss，我觉得用list方便调试，
        step_loss = []  # 可以看到每一步的值和不是全加和到一个变量里
        tk_step = tqdm(self.train_dataloader, desc="EPOCH" + "[TRAIN]" + str(current_epoch),
                       position=0, leave=True, dynamic_ncols=True)
        # position，leave和dynamic_ncols是为了让tqdm显示在最上面

        # 每循环一次应该是一个step
        for _, data in enumerate(tk_step):
            self.step += 1  # step要整个训练过程不断累加，定义成self.
            images, labels = data["image"], data["label"]
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()  # 优化器清零
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()  # 更新模型参数
            total_loss.append(loss.item())
            step_loss.append(loss.item())
            # 下面这一步，在训练进度条后面显示4个衡量指标，因为这一行的作用域仅限于step循环，所以这四个
            #   指标的变量也要设成self. 才能在这个位置访问到
            tk_step.set_postfix({f"train_loss_step": round(loss.item(), 5),
                                 f"train_loss_epoch": round(self.train_loss_last_epoch, 5),
                                 f"val_metric": round(self.val_metric_latest, 5),
                                 f"hvm{self.hvm_epoch}": round(self.highest_val_metric, 5),
                                 f"ltl{self.ltl_epoch}": round(self.lowest_train_loss, 5)})

            # 记录每n个step的平均loss
            if self.step % self.args.log_every_n_step == 0:
                step_ave_loss = round(sum(step_loss) / len(step_loss), 5)
                self.writer.add_scalar("train_loss_step", step_ave_loss, self.step)

        # 记录每个epoch的平均loss
        ave_loss = round(sum(total_loss) / len(total_loss), 5)
        self.writer.add_scalar("train_loss_epoch", ave_loss, current_epoch)

        # 每训练n次保存模型，
        if current_epoch % self.args.save_model_every_n_epoch == 0:
            # 创建新路径
            self.this_model_path = os.path.join(os.path.join(self.writer.log_dir, "checkpoint"),
                                                f"model_{current_epoch}epochs.pth")
            # 保存新模型
            torch.save(self.model.state_dict(), self.this_model_path)

            # 删除旧模型
            if self.previous_model_path is not None and os.path.exists(self.previous_model_path):
                os.remove(self.previous_model_path)
            # 新路径传给旧路径, update
        self.previous_model_path = self.this_model_path

        # 返回这个epoch的loss
        return ave_loss

    def eval_fn(self, current_epoch):
        self.model.eval()
        tk_step = tqdm(self.val_dataloader, desc="EPOCH" + "[VALID]" + str(current_epoch),
                       position=0, leave=True, dynamic_ncols=True)  # Added dynamic_ncols=True

        with torch.no_grad():
            for t, data in enumerate(tk_step):
                images, labels = data["image"], data["label"]
                images, labels = images.to(self.device), labels.to(self.device)

                logits = self.model(images)
                logits = [self.post_pred(i) for i in decollate_batch(logits)]  # 这里用list加后处理没问题
                labels = [self.post_label(i) for i in decollate_batch(labels)]  # 可以运行
                self.metric(logits, labels)
                # 这里好像是可以用的，先不改
                tk_step.set_postfix({"metric": "%4f" % round(self.metric.aggregate().item(), 5)})

            metric = self.metric.aggregate().item()
            self.writer.add_scalar("val_metric", metric, current_epoch)

        return metric

    def train(self):

        self.val_metric_latest = self.eval_fn(0)
        if self.val_metric_latest > self.highest_val_metric:
            self.highest_val_metric = self.val_metric_latest

        td = tqdm(range(self.epochs), desc="Training",
                  position=0, leave=True, dynamic_ncols=True)  # Added dynamic_ncols=True
        # epochs循环
        for i in td:
            current_epoch = i + 1
            train_loss = self.train_fn(current_epoch)
            self.train_loss_last_epoch = train_loss  # 赋给全局变量保存下来

            # 如果新最低，保存这个loss和对应epoch
            if train_loss < self.lowest_train_loss:
                self.lowest_train_loss = train_loss
                self.ltl_epoch = current_epoch

            # 每n个epoch做validation
            if current_epoch % self.args.val_every_n_epoch == 0:

                self.val_metric_latest = self.eval_fn(current_epoch)  # 赋给全局变量保存下来
                print(f"val_metric{current_epoch}={round(self.val_metric_latest, 5)}")

                # 如果新最高，保存这个metric和对应epoch
                if self.val_metric_latest > self.highest_val_metric:
                    self.highest_val_metric = self.val_metric_latest
                    self.hvm_epoch = current_epoch

            self.scheduler.step()  # 学习率按照scheduler的策略更新


def main():
    # writer和checkpoint路径必须要在main()里面声明，我是真的不知道为什么放在里面就没事
    # 放在外面会不断地生成新的文件夹和新的event文件，到底为什么
    # 放在外面是文件作用域，整个文件里的所有调用都直接运行它，所以会生成多个instance，反复创建新文件夹
    # 放在里面是函数作用域，只有调用这个函数时才运行一次，函数没有推出，调用的就一直是同一个instance
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

    # 区分是从0开始还是加载先前训练过的模型
    if args.countiune_train:
        ckpt = torch.load(args.load_model_path)
        # lightning训练出的模型
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
        # pytorch训练的模型
        else:
            model.load_state_dict(ckpt)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150, 300], gamma=0.75)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    metric = DiceMetric(include_background=False, reduction="mean")

    TrainEval(args, model, train_loader, valid_loader, optimizer, scheduler, loss_function, metric, writter,
              device).train()

if __name__ == "__main__":
    main()
