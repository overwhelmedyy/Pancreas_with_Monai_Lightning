
from typing import Tuple

import torch
import torch.nn as nn
from monai.networks.layers import Conv

from monai.networks.nets.basic_unet import UpCat

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Union
import torch.nn.functional as F
from lib.utils.tools.logger import Logger as Log
from lib.models.tools.module_helper import ModuleHelper
from UXNet_3D.networks.UXNet_3D.uxnet_encoder import uxnet_conv

# uxnet_3d + UNetpp
class uxnet_Upp(nn.Module):
    def __init__(self,
                 in_chans=1,
                 out_chans=2,
                 feat_size=(32, 64, 128, 256, 512, 32),

                 hidden_size: int = 512,
                 norm_name: Union[Tuple, str] = "instance",
                 res_block: bool = True,
                 spatial_dims=3,

                 deep_supervision: bool = False,
                 act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                 norm: str | tuple = ("instance", {"affine": True}),
                 bias: bool = True,
                 dropout: float | tuple = 0.0,
                 upsample: str = "deconv",
                 )-> None:

        super().__init__()
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.feat_size = feat_size

        self.out_indice = [] # 就是输出的索引，[0],[1],[2],[3]这些
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.uxnet_3d = uxnet_conv(
            in_chans=self.in_chans,
            depths=[2, 2, 2, 2],
            dims=self.feat_size,
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.upcat_0_1 = UpCat(spatial_dims, self.feat_size[1], self.feat_size[0], self.feat_size[0],
                               act, norm, bias, dropout, upsample, halves=False)
        self.upcat_1_1 = UpCat(spatial_dims, self.feat_size[2], self.feat_size[1], self.feat_size[1],
                               act, norm, bias, dropout, upsample)
        self.upcat_2_1 = UpCat(spatial_dims, self.feat_size[3], self.feat_size[2], self.feat_size[2],
                               act, norm, bias, dropout, upsample)
        self.upcat_3_1 = UpCat(spatial_dims, self.feat_size[4], self.feat_size[3], self.feat_size[3],
                               act, norm, bias, dropout, upsample)

        self.upcat_0_2 = UpCat(spatial_dims, self.feat_size[1], self.feat_size[0] * 2, self.feat_size[0],
                               act, norm, bias, dropout, upsample, halves=False)
        self.upcat_1_2 = UpCat(spatial_dims, self.feat_size[2], self.feat_size[1] * 2, self.feat_size[1],
                               act, norm, bias, dropout, upsample)
        self.upcat_2_2 = UpCat(spatial_dims, self.feat_size[3], self.feat_size[2] * 2, self.feat_size[2],
                               act, norm, bias, dropout, upsample)

        self.upcat_0_3 = UpCat(spatial_dims, self.feat_size[1], self.feat_size[0] * 3, self.feat_size[0],
                               act, norm, bias, dropout, upsample, halves=False)
        self.upcat_1_3 = UpCat(spatial_dims, self.feat_size[2], self.feat_size[1] * 3, self.feat_size[1],
                               act, norm, bias, dropout, upsample)

        self.upcat_0_4 = UpCat(spatial_dims, self.feat_size[1], self.feat_size[0] * 4, self.feat_size[5],
                               act, norm, bias, dropout, upsample, halves=False)

        self.final_conv_0_1 = Conv["conv", spatial_dims](self.feat_size[0], out_chans, kernel_size=1)
        self.final_conv_0_2 = Conv["conv", spatial_dims](self.feat_size[0], out_chans, kernel_size=1)
        self.final_conv_0_3 = Conv["conv", spatial_dims](self.feat_size[0], out_chans, kernel_size=1)
        self.final_conv_0_4 = Conv["conv", spatial_dims](self.feat_size[5], out_chans, kernel_size=1)



    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)
        enc1 = self.encoder1(x_in)  # x_in 1 1 64 64 64, enc1 1 32 64 64 64
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])

        x_0_0 = enc1
        x_1_0 = enc2
        x_2_0 = enc3
        x_3_0 = enc4
        x_4_0 = enc_hidden

        x_0_1 = self.upcat_0_1(x_1_0, x_0_0)
        x_1_1 = self.upcat_1_1(x_2_0, x_1_0)
        x_2_1 = self.upcat_2_1(x_3_0, x_2_0)
        x_3_1 = self.upcat_3_1(x_4_0, x_3_0)

        x_0_2 = self.upcat_0_2(x_1_1, torch.cat([x_0_0, x_0_1], dim=1))
        x_1_2 = self.upcat_1_2(x_2_1, torch.cat([x_1_0, x_1_1], dim=1))
        x_2_2 = self.upcat_2_2(x_3_1, torch.cat([x_2_0, x_2_1], dim=1))

        x_0_3 = self.upcat_0_3(x_1_2, torch.cat([x_0_0, x_0_1, x_0_2], dim=1))
        x_1_3 = self.upcat_1_3(x_2_2, torch.cat([x_1_0, x_1_1, x_1_2], dim=1))

        x_0_4 = self.upcat_0_4(x_1_3, torch.cat([x_0_0, x_0_1, x_0_2, x_0_3], dim=1))

        output_0_1 = self.final_conv_0_1(x_0_1)
        output_0_2 = self.final_conv_0_2(x_0_2)
        output_0_3 = self.final_conv_0_3(x_0_3)
        output_0_4 = self.final_conv_0_4(x_0_4)

        output = [output_0_4]

        return output

    def para_num(network):
        return sum(p.numel() for p in network.parameters())


if __name__ == "__main__":
    module = uxnet_Upp()
    print(module.para_num())
    x = torch.rand(1, 1, 64, 64, 64)
    y = module(x)
    print(y[0].shape)