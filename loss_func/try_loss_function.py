# import pickle
#
# import numpy as np
# from torch import nn
#
# from loss_func.SegLossOdyssey.losses_pytorch.dice_loss import DC_and_topk_loss, DC_and_CE_loss, softmax_helper, get_tp_fp_fn
# import torch
# import os
#
# from loss_func.SegLossOdyssey.losses_pytorch.ND_Crossentropy import CrossentropyND
#
# torch.set_float32_matmul_precision("medium")
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
#
# with open(r'C:\Git\NeuralNetwork\Pancreas_with_Monai_Lightning\loss_func\pickle_model_output\label1.pickle', 'rb') as file:
#     # Load the data from the file
#     label1 = pickle.load(file)
#
with open(r'C:\Git\NeuralNetwork\Pancreas_with_Monai_Lightning\loss_func\pickle_model_output\logit1.pickle', 'rb') as file:
        logit1 = pickle.load(file)
logit1_cpu = logit1[0][0][1][:,:,32].cpu()
logit1_numpy = logit1_cpu.numpy()
plt.imshow(logit1_numpy)
plt.show()
#
# class TopKLoss(CrossentropyND):
#     """
#     Network has to have NO LINEARITY!
#     """
#     def __init__(self, weight=None, ignore_index=-100, k=10):
#         self.k = k
#         super(TopKLoss, self).__init__(weight, None, ignore_index, reduce=False)
#
#     def forward(self, inp, target):
#         target = target[:, 0].long()
#         res = super(TopKLoss, self).forward(inp, target)
#         num_voxels = np.prod(res.shape)
#         res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
#         return res.mean()
#
# class SoftDiceLoss(nn.Module):
#     def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
#                  square=False):
#         """
#         paper: https://arxiv.org/pdf/1606.04797.pdf
#         """
#         super(SoftDiceLoss, self).__init__()
#
#         self.square = square
#         self.do_bg = do_bg
#         self.batch_dice = batch_dice
#         self.apply_nonlin = apply_nonlin
#         self.smooth = smooth
#
#     def forward(self, x, y, loss_mask=None):
#         shp_x = x.shape
#
#         if self.batch_dice:
#             axes = [0] + list(range(2, len(shp_x)))
#         else:
#             axes = list(range(2, len(shp_x)))
#
#         if self.apply_nonlin is not None:
#             x = self.apply_nonlin(x)
#
#         tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)
#
#         dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
#
#         if not self.do_bg:
#             if self.batch_dice:
#                 dc = dc[1:]
#             else:
#                 dc = dc[:, 1:]
#         dc = dc.mean()
#
#         return -dc
# class DTK_loss(DC_and_topk_loss):
#     def __init__(self, ce_kwargs1, ce_kwargs2, soft_dice_kwargs1, soft_dice_kwargs2, soft_dice_kwargs3, aggregate="sum"):
#         super(DC_and_topk_loss, self).__init__()
#         self.aggregate = aggregate
#         self.ce1 = TopKLoss(**ce_kwargs1)
#         self.ce2 = TopKLoss(**ce_kwargs2)
#         self.dc1 = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs1)
#         self.dc2 = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs2)
#         self.dc3 = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs3)
#
#     def forward(self, net_output, target):
#         ce_loss1 = self.ce1(net_output[0], target)
#         ce_loss2 = self.ce2(net_output[1], target)
#         dc_loss1 = self.dc1(net_output[2], target)
#         dc_loss2 = self.dc2(net_output[3], target)
#         dc_loss3 = self.dc3(net_output[4], target)
#         if self.aggregate == "sum":
#             result = ce_loss1 + ce_loss2 + dc_loss1 + dc_loss2 + dc_loss3
#         else:
#             raise NotImplementedError("nah son") # reserved for other stuff (later?)
#         return result





