from typing import Tuple, Sequence

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import LayerNorm

from monai.networks.blocks import downsample, PatchEmbed, Convolution
from monai.networks.layers import Conv, Norm, DropPath

from monai.networks.nets.basic_unet import UpCat


from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Union
import torch.nn.functional as F

from monai.networks.nets.swin_unetr import MERGING_MODE, BasicLayer
from monai.utils import look_up_option

def proj_out(x, normalize=False):
    if normalize:
        x_shape = x.size()
        if len(x_shape) == 5:
            n, ch, d, h, w = x_shape
            x = rearrange(x, "n c d h w -> n d h w c")
            x = F.layer_norm(x, [ch])
            x = rearrange(x, "n d h w c -> n c d h w")
        elif len(x_shape) == 4:
            n, ch, h, w = x_shape
            x = rearrange(x, "n c h w -> n h w c")
            x = F.layer_norm(x, [ch])
            x = rearrange(x, "n h w c -> n c h w")
    return x


class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            in_chans: int,
            embed_dim: int,
            window_size: Sequence[int],
            patch_size: Sequence[int],
            depths: Sequence[int],
            num_heads: Sequence[int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            norm_layer: type[LayerNorm] = nn.LayerNorm,
            patch_norm: bool = False,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
            downsample="merging",
            use_v2=False,
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beginning of each swin stage.
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.use_v2 = use_v2
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        if self.use_v2:
            self.layers1c = nn.ModuleList()
            self.layers2c = nn.ModuleList()
            self.layers3c = nn.ModuleList()
            self.layers4c = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
            if self.use_v2:
                layerc1 = ux_block(dim=36)
                layerc2 = ux_block(dim=72)
                layerc3 = ux_block(dim=144)
                layerc4 = ux_block(dim=288)
                if i_layer == 0:
                    self.layers1c.append(layerc1)
                elif i_layer == 1:
                    self.layers2c.append(layerc2)
                elif i_layer == 2:
                    self.layers3c.append(layerc3)
                elif i_layer == 3:
                    self.layers4c.append(layerc4)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = proj_out(x0, normalize)
        if self.use_v2:
            x0 = self.layers1c[0](x0.contiguous())
        x1 = self.layers1[0](x0.contiguous())
        x1_out = proj_out(x1, normalize)
        if self.use_v2:
            x1 = self.layers2c[0](x1.contiguous())
        x2 = self.layers2[0](x1.contiguous())
        x2_out = proj_out(x2, normalize)
        if self.use_v2:
            x2 = self.layers3c[0](x2.contiguous())
        x3 = self.layers3[0](x2.contiguous())
        x3_out = proj_out(x3, normalize)
        if self.use_v2:
            x3 = self.layers4c[0](x3.contiguous())
        x4 = self.layers4[0](x3.contiguous())
        x4_out = proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]

class ux_block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        # self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, groups=dim)
        self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(4 * dim, dim)
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, groups=dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = x .permute(0, 4, 1, 2, 3)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 3, 4, 1)
        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 4, 1, 2, 3)
        x = input + x
        return x

    def para_num(self):
        for name, block in self.named_children():
            params = sum(p.numel() for p in block.parameters())
            print(f"{name}: {round(params / 1e6, 4)}M")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total: {round(total_params/ 1e6, 4)}M")


# uxnet_3d + UNetpp
class SwinViT_uxblock_Upp(nn.Module):
    def __init__(self,
                 in_chans=1,
                 out_channels=2,
                 depths=(2, 2, 2, 2),
                 feat_size=(18, 36, 72, 144, 288, 576, 18),

                 norm_name: Union[Tuple, str] = "instance",

                 res_block: bool = True,
                 spatial_dims=3,


                 act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                 norm: str | tuple = ("instance", {"affine": True}),
                 bias: bool = True,
                 dropout: float | tuple = 0.0,
                 upsample: str = "deconv",

                 normalize: bool = True,
                 downsample="merging",
                 ) -> None:
        super().__init__()

        self.swinViT = SwinTransformer(
            in_chans=in_chans,
            embed_dim=feat_size[1],  # 默认值24，一般设成48
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=depths,
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=True,
        )

        self.normalize = normalize

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[1],
            out_channels=feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[2],
            out_channels=feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[3],
            out_channels=feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[4],
            out_channels=feat_size[4],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[5],
            out_channels=feat_size[5],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.upcat_0_1 = UpCat(spatial_dims, feat_size[1], feat_size[0], feat_size[0],
                               act, norm, bias, dropout, upsample, halves=False)
        self.upcat_1_1 = UpCat(spatial_dims, feat_size[2], feat_size[1], feat_size[1],
                               act, norm, bias, dropout, upsample)
        self.upcat_2_1 = UpCat(spatial_dims, feat_size[3], feat_size[2], feat_size[2],
                               act, norm, bias, dropout, upsample)
        self.upcat_3_1 = UpCat(spatial_dims, feat_size[4], feat_size[3], feat_size[3],
                               act, norm, bias, dropout, upsample)
        self.upcat_4_1 = UpCat(spatial_dims, feat_size[5], feat_size[4], feat_size[4],
                               act, norm, bias, dropout, upsample)

        self.upcat_0_2 = UpCat(spatial_dims, feat_size[1], feat_size[0] * 2, feat_size[0],
                               act, norm, bias, dropout, upsample, halves=False)
        self.upcat_1_2 = UpCat(spatial_dims, feat_size[2], feat_size[1] * 2, feat_size[1],
                               act, norm, bias, dropout, upsample)
        self.upcat_2_2 = UpCat(spatial_dims, feat_size[3], feat_size[2] * 2, feat_size[2],
                               act, norm, bias, dropout, upsample)
        self.upcat_3_2 = UpCat(spatial_dims, feat_size[4], feat_size[3] * 2, feat_size[3],
                               act, norm, bias, dropout, upsample)

        self.upcat_0_3 = UpCat(spatial_dims, feat_size[1], feat_size[0] * 3, feat_size[0],
                               act, norm, bias, dropout, upsample, halves=False)
        self.upcat_1_3 = UpCat(spatial_dims, feat_size[2], feat_size[1] * 3, feat_size[1],
                               act, norm, bias, dropout, upsample)
        self.upcat_2_3 = UpCat(spatial_dims, feat_size[3], feat_size[2] * 3, feat_size[2],
                               act, norm, bias, dropout, upsample)

        self.upcat_0_4 = UpCat(spatial_dims, feat_size[1], feat_size[0] * 4, feat_size[0],
                               act, norm, bias, dropout, upsample, halves=False)
        self.upcat_1_4 = UpCat(spatial_dims, feat_size[2], feat_size[1] * 4, feat_size[1],
                               act, norm, bias, dropout, upsample)

        self.upcat_0_5 = UpCat(spatial_dims, feat_size[1], feat_size[0] * 5, feat_size[0],
                               act, norm, bias, dropout, upsample, halves=False)

        self.final_conv_0_1 = Conv["conv", spatial_dims](feat_size[0], out_channels, kernel_size=1)
        self.final_conv_0_2 = Conv["conv", spatial_dims](feat_size[0], out_channels, kernel_size=1)
        self.final_conv_0_3 = Conv["conv", spatial_dims](feat_size[0], out_channels, kernel_size=1)
        self.final_conv_0_4 = Conv["conv", spatial_dims](feat_size[0], out_channels, kernel_size=1)
        self.final_conv_0_5 = Conv["conv", spatial_dims](feat_size[0], out_channels, kernel_size=1)

    def forward(self, x_in):
        outs = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder0(x_in)
        enc1 = self.encoder1(outs[0])
        enc2 = self.encoder2(outs[1])
        enc3 = self.encoder3(outs[2])
        enc4 = self.encoder4(outs[3])
        enc5 = self.encoder5(outs[4])

        x_0_0 = enc0
        x_1_0 = enc1
        x_2_0 = enc2
        x_3_0 = enc3
        x_4_0 = enc4
        x_5_0 = enc5

        x_0_1 = self.upcat_0_1(x_1_0, x_0_0)
        x_1_1 = self.upcat_1_1(x_2_0, x_1_0)
        x_2_1 = self.upcat_2_1(x_3_0, x_2_0)
        x_3_1 = self.upcat_3_1(x_4_0, x_3_0)
        x_4_1 = self.upcat_4_1(x_5_0, x_4_0)

        x_0_2 = self.upcat_0_2(x_1_1, torch.cat([x_0_0, x_0_1], dim=1))
        x_1_2 = self.upcat_1_2(x_2_1, torch.cat([x_1_0, x_1_1], dim=1))
        x_2_2 = self.upcat_2_2(x_3_1, torch.cat([x_2_0, x_2_1], dim=1))
        x_3_2 = self.upcat_3_2(x_4_1, torch.cat([x_3_0, x_3_1], dim=1))

        x_0_3 = self.upcat_0_3(x_1_2, torch.cat([x_0_0, x_0_1, x_0_2], dim=1))
        x_1_3 = self.upcat_1_3(x_2_2, torch.cat([x_1_0, x_1_1, x_1_2], dim=1))
        x_2_3 = self.upcat_2_3(x_3_2, torch.cat([x_2_0, x_2_1, x_2_2], dim=1))

        x_0_4 = self.upcat_0_4(x_1_3, torch.cat([x_0_0, x_0_1, x_0_2, x_0_3], dim=1))
        x_1_4 = self.upcat_1_4(x_2_3, torch.cat([x_1_0, x_1_1, x_1_2, x_1_3], dim=1))

        x_0_5 = self.upcat_0_5(x_1_4, torch.cat([x_0_0, x_0_1, x_0_2, x_0_3, x_0_4], dim=1))

        output_0_1 = self.final_conv_0_1(x_0_1)
        output_0_2 = self.final_conv_0_2(x_0_2)
        output_0_3 = self.final_conv_0_3(x_0_3)
        output_0_4 = self.final_conv_0_4(x_0_4)
        output_0_5 = self.final_conv_0_5(x_0_5)

        output = [output_0_1, output_0_2, output_0_3, output_0_4, output_0_5]
        # output = output_0_5

        return output

    def para_num(self):
        for name, block in self.named_children():
            params = sum(p.numel() for p in block.parameters())
            print(f"{name}: {round(params / 1e6, 4)}M")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total: {round(total_params/ 1e6, 4)}M")





if __name__ == "__main__":
    module1 = SwinViT_uxblock_Upp()
    # module1.para_num()
    x = torch.rand(1, 1, 128, 128, 128)
    y = module1(x)
    print(y[0].shape)
