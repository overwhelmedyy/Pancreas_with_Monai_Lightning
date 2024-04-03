import argparse
import glob
import os
import random
from datetime import datetime
import torch.nn.functional as F
import numpy as np
import torch
from torch.hub import tqdm
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

from monai.data import pad_list_data_collate, decollate_batch, DataLoader, CacheDataset
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose, LoadImaged, RandCropByPosNegLabeld,
    RandAffined, RandRotated, RandFlipd, ResizeWithPadOrCropd, EnsureChannelFirstd, EnsureType, AsDiscrete
)
from my_network.SwinViT_Upp import SwinViT_Upp
from my_network.SwinViT_Upp_attg import SwinViT_Upp_attg

args = argparse.Namespace(
    task_name="Task01_pancreas",
    network_name="SwinViT_Upp_pytorch",
    countiune_train=True,  # True ot False
    load_model_path=r"C:\Git\NeuralNetwork\Pancreas_with_Monai_Lightning\runs\Task01_pancreas\SwinViT_Upp_pytorch\best260_0.8577467203140259.pth",
    no_cuda=False,  # disables CUDA training
    save_model_every_n_epoch=1,  # save model every epochs (default : 1)
    val_every_n_epoch=5,  # validate every epochs (default : 1)
    log_every_n_step=10,  # log every step (default : 1)
    epochs=1000,  # number of epochs (default : 10)
    lr=2.5e-4,  # base learning rate (default : 0.01)
    batch_size=1,  # batch size (default : 4)
    dry_run=False  # quickly check a single pass
)

time_stamp = datetime.now().strftime('%m%d_%H%M')

directory = os.environ.get("MONAI_DATA_DIRECTORY")
data_dir = os.path.join(directory, args.task_name)
persistent_cache = os.path.join(data_dir, "persistent_cache")
cuda = torch.device("cuda:0")

class SLL_Loss():

    def __init__(self):
        self.DCLoss = DiceCELoss(to_onehot_y=True, softmax=True, lambda_ce=0.0, lambda_dice=1.0)
        self.CELoss = DiceCELoss(to_onehot_y=True, softmax=True, lambda_ce=1.0, lambda_dice=0.0)
        self.MSELoss = MSELoss()

    def dist2(self, candidates, prototype):
        x = F.normalize(candidates, dim=0, p=2).permute(1, 0).unsqueeze(0)
        y = F.normalize(prototype, dim=0, p=2).permute(1, 0).unsqueeze(0)
        # Computes batched the p-norm distance between each pair of the two collections of row vectors
        # 求的是p-norm距离，p=2就是欧式距离
        loss = torch.cdist(x, y, p=2.0).mean()

        return loss

    def compute_uxi_loss(self, predicta, predictb, represent_a, percent=20):
        # predicta和predictb是网络a和网络b的输出logit，represent_a是网络a的特征，也就是a的倒数第二层的输出，channel=18的那个
        batch_size, num_class, h, w, d = predicta.shape

        # 找出预测中的最大值及其坐标
        # 得到的logits_u_a就相当于是把onehot压平的logtis
        logits_u_a, index_u_a = torch.max(predicta, dim=1)  # logits是data，index是坐标position
        logits_u_b, index_u_b = torch.max(predictb, dim=1)  # logits后面好像一直都没有用到
        target = index_u_a | index_u_b

        ## 做mask
        with torch.no_grad():
            # drop pixels with high entropy from a
            # 应该是在求熵的时候，求出的是一个类似于特征平面的东西，在这个平面上求L2距离
            # 最小的那20%视为unreliable的predict，用他们的位置作mask

            entropy_a = -torch.sum(predicta * torch.log(predicta + 1e-10), dim=1)
            thresh_a = np.percentile(entropy_a.detach().cpu().numpy().flatten(), percent)
            thresh_mask_a = entropy_a.ge(thresh_a).bool()

            # drop pixels with high entropy from b
            entropy_b = -torch.sum(predictb * torch.log(predictb + 1e-10), dim=1)
            thresh_b = np.percentile(entropy_b.detach().cpu().numpy().flatten(), percent)
            thresh_mask_b = entropy_b.ge(thresh_b).bool()

            thresh_mask = torch.logical_and(thresh_mask_a, thresh_mask_b)

            # 如果一个voxel即是a的高熵点，也是b的高熵点，那么就把它标记为2
            # thresh_mask全是True/False值，作为mask
            target[thresh_mask] = 2
            target_clone = torch.clone(target.view(-1))

            represent_a = represent_a.permute(1, 0, 2, 3, 4)
            # print(represent_a.size())
            # 下面是reliable pred，prototypes for each category using the reliable set as a base
            # 下面的两个就是prototype
            represent_a = represent_a.contiguous().view(represent_a.size(0), -1)
            # 下面两个prototype是形状都是(2,)，mean(dim=1)求了两个平均值
            prototype_f = represent_a[:, target_clone == 1].mean(dim=1)  # target=1是前景，foreground
            prototype_b = represent_a[:, target_clone == 0].mean(dim=1)  # target=0是背景，background

            # 下面是unreliable pred，=1就是潜在的前景，=0就是潜在的背景
            # foreground_candidate形状是（2，575447）
            candidate_f = represent_a[:, (target_clone == 2) & (index_u_a.view(-1) == 1)]
            candidate_b = represent_a[:, (target_clone == 2) & (index_u_a.view(-1) == 0)]

            # num_samples=5754,从前景中取5754个点，背景中也取这么多
            num_samples = candidate_f.size(1) // 100

            # randperm生成从0到5754的整数，随机打乱的排列，然后取num_samples这么多个sample
            selected_indices_f = torch.randperm(candidate_f.size(1))[:num_samples]
            selected_indices_b = torch.randperm(candidate_b.size(1))[:num_samples]

            # contrastive_loss就是L2约束项，也就是L_seg
            # 分别对前景和背景的unreliable voxel到相应prototype的L2距离进行约束
            contrastive_loss_f = self.dist2(candidate_f[:, selected_indices_f], prototype_f.unsqueeze(dim=1))
            contrastive_loss_b = self.dist2(candidate_b[:, selected_indices_b], prototype_b.unsqueeze(dim=1))
            # 再求两个prototype的L2距离
            contrastive_loss_c = self.dist2(prototype_f.unsqueeze(dim=1), prototype_b.unsqueeze(dim=1))
            # 总的contrastive loss是把这三项加起来
            con_loss = contrastive_loss_f + contrastive_loss_b + contrastive_loss_c

            weight = batch_size * h * w * d / torch.sum(target != 2)

        loss_a = weight * F.cross_entropy(predicta, target, ignore_index=2)  # 把加进去的index=2忽略掉
        loss_b = weight * F.cross_entropy(predictb, target, ignore_index=2)

        return loss_a, loss_b, con_loss

    def forward(self, output, label):
        output_soft5 = F.softmax(output[5], dim=1)
        ce_loss5 = self.CELoss(output[5], label)
        dice_loss5 = self.DCLoss(output[5], label)

        output_soft4 = F.softmax(output[4], dim=1)
        ce_loss4 = self.CELoss(output[4], label)
        dice_loss4 = self.DCLoss(output[4], label)

        output_soft3 = F.softmax(output[3], dim=1)
        ce_loss3 = self.CELoss(output[3], label)
        dice_loss3 = self.DCLoss(output[3], label)

        output_soft2 = F.softmax(output[2], dim=1)
        ce_loss2 = self.CELoss(output[2], label)
        dice_loss2 = self.DCLoss(output[2], label)

        ## Cross reliable loss term

        probability_5, index_5 = torch.max(output_soft5, dim=1)
        probability_4, index_4 = torch.max(output_soft4, dim=1)
        conf_diff_mask = (
                ((index_5 == 1) & (probability_5 >= 0.6)) ^ ((index_4 == 1) & (probability_4 >= 0.6))).to(
            torch.int32)

        mse_dist5 = self.MSELoss(output_soft5[:, 1, ...], label[:, 0, ...])
        mse_dist4 = self.MSELoss(output_soft4[:, 1, ...], label[:, 0, ...])
        mse_dist3 = self.MSELoss(output_soft3[:, 1, ...], label[:, 0, ...])
        mse_dist2 = self.MSELoss(output_soft2[:, 1, ...], label[:, 0, ...])

        mistake5 = torch.sum(conf_diff_mask * mse_dist5) / (torch.sum(conf_diff_mask) + 1e-16)
        mistake4 = torch.sum(conf_diff_mask * mse_dist4) / (torch.sum(conf_diff_mask) + 1e-16)
        mistake3 = torch.sum(conf_diff_mask * mse_dist3) / (torch.sum(conf_diff_mask) + 1e-16)
        mistake2 = torch.sum(conf_diff_mask * mse_dist2) / (torch.sum(conf_diff_mask) + 1e-16)

        supervised_loss5 = (ce_loss5 + dice_loss5) + 0.5 * mistake5
        supervised_loss4 = (ce_loss4 + dice_loss4) + 0.5 * mistake4
        supervised_loss3 = (ce_loss3 + dice_loss3) + 0.5 * mistake3
        supervised_loss2 = (ce_loss2 + dice_loss2) + 0.5 * mistake2

        outputs_clone5 = output_soft5.clone().detach()
        outputs_clone4 = output_soft4.clone().detach()
        outputs_clone3 = output_soft3.clone().detach()
        outputs_clone2 = output_soft2.clone().detach()

        loss_u_5, loss_u_2, con_loss5 = self.compute_uxi_loss(outputs_clone5, outputs_clone2, outputs_clone5, percent=20)
        loss_u_4, loss_u_3, con_loss4 = self.compute_uxi_loss(outputs_clone4, outputs_clone3, outputs_clone4, percent=20)
        loss5 = supervised_loss5 + loss_u_5 + 0.2 * con_loss5  # con_loss加到dice大的网络上
        loss4 = supervised_loss4 + loss_u_4 + 0.2 * con_loss4
        loss3 = supervised_loss3 + loss_u_3
        loss2 = supervised_loss2 + loss_u_2

        total_loss = loss5 + loss4 + loss3 + loss2
        return total_loss

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

        self.best_model_path = None  # 保存最好的模型的路径
        self.nd_model_path = None  # 删除第二好的模型

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
                logits = [self.post_pred(i) for i in decollate_batch(logits[4])]  # 这里用list加后处理没问题
                labels = [self.post_label(i) for i in decollate_batch(labels)]  # 可以运行
                self.metric(logits, labels)
                # 这里好像是可以用的，先不改
                tk_step.set_postfix({"metric": "%4f" % round(self.metric.aggregate().item(), 5)})

            metric = self.metric.aggregate().item()
            self.metric.reset()
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
                    self.nd_model_path = self.best_model_path
                    self.best_model_path = os.path.join(self.writer.log_dir, "checkpoint",
                                                        f"best{self.hvm_epoch}_{self.highest_val_metric}.pth")

                    torch.save(self.model.state_dict(), self.best_model_path)
                    if self.nd_model_path is not None and os.path.exists(self.nd_model_path):
                        os.remove(self.nd_model_path)

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
                pos=3,
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

    # model = SwinUNETR(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=2,
    #     img_size=(64, 64, 64)
    # )

    model = SwinViT_Upp()

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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 400], gamma=1)
    # loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    metric = DiceMetric(include_background=False, reduction="mean")
    my_ssl_loss = SLL_Loss()

    TrainEval(args, model, train_loader, valid_loader, optimizer, scheduler, my_ssl_loss, metric, writter,
              device).train()

if __name__ == "__main__":
    main()
