# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding
from monai.networks.layers import Conv, trunc_normal_
from monai.utils import deprecated_arg, ensure_tuple_rep, optional_import
from monai.utils.module import look_up_option

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_PATCH_EMBEDDING_TYPES = {"conv", "perceptron"}
SUPPORTED_POS_EMBEDDING_TYPES = {"none", "learnable", "sincos"}


class PatchEmbeddingBlock(nn.Module):
    """
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Example::

        >>> from monai.networks.blocks import PatchEmbeddingBlock
        >>> PatchEmbeddingBlock(in_channels=4, img_size=32, patch_size=8, hidden_size=32, num_heads=4,
        >>>                     proj_type="conv", pos_embed_type="sincos")

    """

    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int,
        num_heads: int,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            proj_type: patch embedding layer type.
            pos_embed_type: position embedding layer type.
            dropout_rate: fraction of the input units to drop.
            spatial_dims: number of spatial dimensions.
        .. deprecated:: 1.4
            ``pos_embed`` is deprecated in favor of ``proj_type``.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError(f"dropout_rate {dropout_rate} should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden size {hidden_size} should be divisible by num_heads {num_heads}.")

        self.proj_type = look_up_option(proj_type, SUPPORTED_PATCH_EMBEDDING_TYPES)
        self.pos_embed_type = look_up_option(pos_embed_type, SUPPORTED_POS_EMBEDDING_TYPES)

        # img_size是边长，补全成正方形（sptial_dims=2）或立方体（sptial_dims=3）
        img_size = ensure_tuple_rep(img_size, spatial_dims) # img_size = (224, 224, 224)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims) # patch_size = (16, 16, 16)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.proj_type == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        # 一共有几个patch
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)]) # 2744
        # 每个patch有几个像素，这是patch的dim
        self.patch_dim = int(in_channels * np.prod(patch_size)) # 4096

        self.patch_embeddings: nn.Module
        if self.proj_type == "conv":
            self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            )
        elif self.proj_type == "perceptron":
            # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}
            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len), nn.Linear(self.patch_dim, hidden_size)
            )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size)) # (1,2744,4096)
        self.dropout = nn.Dropout(dropout_rate)

        if self.pos_embed_type == "none":
            pass
        elif self.pos_embed_type == "learnable":
            trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        elif self.pos_embed_type == "sincos":
            grid_size = []
            for in_size, pa_size in zip(img_size, patch_size):
                grid_size.append(in_size // pa_size) #[14, 14, 14]

            with torch.no_grad():
                pos_embeddings = build_sincos_position_embedding(grid_size, hidden_size, spatial_dims)
                self.position_embeddings.data.copy_(pos_embeddings.float())
        else:
            raise ValueError(f"pos_embed_type {self.pos_embed_type} not supported.")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embeddings(x)
        if self.proj_type == "conv":
            x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class PatchEmbed(nn.Module):
    """
    Patch embedding block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Unlike ViT patch embedding block: (1) input is padded to satisfy window size requirements (2) normalized if
    specified (3) position embedding is not used.

    Example::

        >>> from monai.networks.blocks import PatchEmbed
        >>> PatchEmbed(patch_size=2, in_chans=1, embed_dim=48, norm_layer=nn.LayerNorm, spatial_dims=3)
    """

    def __init__(
        self,
        patch_size: Sequence[int] | int = 2,
        in_chans: int = 1,
        embed_dim: int = 48,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            patch_size: dimension of patch size.
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            norm_layer: normalization layer.
            spatial_dims: spatial dimension.
        """

        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = Conv[Conv.CONV, spatial_dims](
            in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            _, _, d, h, w = x_shape
            if w % self.patch_size[2] != 0:
                x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
            if h % self.patch_size[1] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
            if d % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))

        elif len(x_shape) == 4:
            _, _, h, w = x_shape
            if w % self.patch_size[1] != 0:
                x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))
            if h % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            if len(x_shape) == 5:
                d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
                x = x.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
            elif len(x_shape) == 4:
                wh, ww = x_shape[2], x_shape[3]
                x = x.transpose(1, 2).view(-1, self.embed_dim, wh, ww)
        return x


if __name__ == "__main__":
    import torch

    # test PatchEmbeddingBlock
    patch_embed = PatchEmbeddingBlock(
        in_channels=1, img_size=224, patch_size=16, hidden_size=4096, num_heads=16, proj_type="conv", pos_embed_type="sincos"
    )
    print(patch_embed)

    # test PatchEmbed
    # patch_embed = PatchEmbed(patch_size=2, in_chans=1, embed_dim=48, norm_layer=nn.LayerNorm, spatial_dims=3)
    # print(patch_embed)
    # x = torch.randn(2, 1, 30, 31, 29)
    # print(patch_embed(x))
