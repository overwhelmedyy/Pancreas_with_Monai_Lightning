import os
import pickle
from monai.losses import DiceCELoss
import numpy as np
import torch
import torch.nn.functional as F

from loss_func.SSL_contrastive.src.train_LA import get_current_consistency_weight
from loss_func.SSL_contrastive.src.utils import ramps, losses
from monai.losses import DiceLoss, DiceCELoss
from torch.nn import MSELoss

# def compute_uxi_loss(predicta, predictb, represent_a, percent=20):
#     # predicta和predictb是网络a和网络b的输出logit，represent_a是网络a的特征，也就是a的倒数第二层的输出，channel=18的那个
#     batch_size, num_class, h, w, d = predicta.shape
#
#     # 找出预测中的最大值及其坐标
#     # 得到的logits_u_a就相当于是把onehot压平的logtis
#     logits_u_a, index_u_a = torch.max(predicta, dim=1)  # logits是data，index是坐标position
#     logits_u_b, index_u_b = torch.max(predictb, dim=1)  # logits后面好像一直都没有用到
#     target = index_u_a | index_u_b
#
#     ## 做mask
#     with torch.no_grad():
#         # drop pixels with high entropy from a
#         # 应该是在求熵的时候，求出的是一个类似于特征平面的东西，在这个平面上求L2距离
#         # 最小的那20%视为unreliable的predict，用他们的位置作mask
#
#         entropy_a = -torch.sum(predicta * torch.log(predicta + 1e-10), dim=1)
#         thresh_a = np.percentile(entropy_a.detach().cpu().numpy().flatten(), percent)
#         thresh_mask_a = entropy_a.ge(thresh_a).bool()
#
#         # drop pixels with high entropy from b
#         entropy_b = -torch.sum(predictb * torch.log(predictb + 1e-10), dim=1)
#         thresh_b = np.percentile(entropy_b.detach().cpu().numpy().flatten(), percent)
#         thresh_mask_b = entropy_b.ge(thresh_b).bool()
#
#         thresh_mask = torch.logical_and(thresh_mask_a, thresh_mask_b)
#
#         # 如果一个voxel即是a的高熵点，也是b的高熵点，那么就把它标记为2
#         # thresh_mask全是True/False值，作为mask
#         target[thresh_mask] = 2
#         target_clone = torch.clone(target.view(-1))
#
#         represent_a = represent_a.permute(1, 0, 2, 3, 4)
#         # print(represent_a.size())
#         # 下面是reliable pred，prototypes for each category using the reliable set as a base
#         # 下面的两个就是prototype
#         represent_a = represent_a.contiguous().view(represent_a.size(0), -1)
#         # 下面两个prototype是形状都是(2,)，mean(dim=1)求了两个平均值
#         prototype_f = represent_a[:, target_clone == 1].mean(dim=1)  # target=1是前景，foreground
#         prototype_b = represent_a[:, target_clone == 0].mean(dim=1)  # target=0是背景，background
#
#         # 下面是unreliable pred，=1就是潜在的前景，=0就是潜在的背景
#         # foreground_candidate形状是（2，575447）
#         candidate_f = represent_a[:, (target_clone == 2) & (index_u_a.view(-1) == 1)]
#         candidate_b = represent_a[:, (target_clone == 2) & (index_u_a.view(-1) == 0)]
#
#         # num_samples=5754,从前景中取5754个点，背景中也取这么多
#         num_samples = candidate_f.size(1) // 100
#
#         # randperm生成从0到5754的整数，随机打乱的排列，然后取num_samples这么多个sample
#         selected_indices_f = torch.randperm(candidate_f.size(1))[:num_samples]
#         selected_indices_b = torch.randperm(candidate_b.size(1))[:num_samples]
#
#         # contrastive_loss就是L2约束项，也就是L_seg
#         # 分别对前景和背景的unreliable voxel到相应prototype的L2距离进行约束
#         contrastive_loss_f = dist2(candidate_f[:, selected_indices_f], prototype_f.unsqueeze(dim=1))
#         contrastive_loss_b = dist2(candidate_b[:, selected_indices_b], prototype_b.unsqueeze(dim=1))
#         # 再求两个prototype的L2距离
#         contrastive_loss_c = dist2(prototype_f.unsqueeze(dim=1), prototype_b.unsqueeze(dim=1))
#         # 总的contrastive loss是把这三项加起来
#         con_loss = contrastive_loss_f + contrastive_loss_b + contrastive_loss_c
#
#         weight = batch_size * h * w * d / torch.sum(target != 2)
#
#     loss_a = weight * F.cross_entropy(predicta, target, ignore_index=2)  # 把加进去的index=2忽略掉
#     loss_b = weight * F.cross_entropy(predictb, target, ignore_index=2)
#
#     return loss_a, loss_b, con_loss

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

    def forward(self, output5, output4, output3, output2, label):
        output_soft5 = F.softmax(output5, dim=1)
        ce_loss5 = self.CELoss(output5, label)
        dice_loss5 = self.DCLoss(output5, label)

        output_soft4 = F.softmax(output4, dim=1)
        ce_loss4 = self.CELoss(output4, label)
        dice_loss4 = self.DCLoss(output4, label)

        output_soft3 = F.softmax(output3, dim=1)
        ce_loss3 = self.CELoss(output3, label)
        dice_loss3 = self.DCLoss(output3, label)

        output_soft2 = F.softmax(output2, dim=1)
        ce_loss2 = self.CELoss(output2, label)
        dice_loss2 = self.DCLoss(output2, label)

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


if __name__ == '__main__':
    torch.set_float32_matmul_precision("medium")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    with open(r'C:\Git\NeuralNetwork\Pancreas_with_Monai_Lightning\loss_func\pickle_model_output\label1.pickle',
              'rb') as file:
        # Load the data from the file
        label1 = pickle.load(file)

    with open(r'C:\Git\NeuralNetwork\Pancreas_with_Monai_Lightning\loss_func\pickle_model_output\logit1.pickle',
              'rb') as file:
        # Load the data from the file
        logit1 = pickle.load(file)

    labeled_bs = 2
    Thresh = 0.5
    consistency_criterion = torch.nn.MSELoss()
    iter_num = 1000
    # my_sll_loss(v_outputs, r_outputs, v_label, r_label, v_rep, r_rep, labeled_bs, Thresh, consistency_criterion, iter_num)
    a, b = my_sll_loss(label, logit1[4], logit1[1], label1, logit1[3], logit1[0], 0.7, consistency_criterion, 200)
    print(a, b)
