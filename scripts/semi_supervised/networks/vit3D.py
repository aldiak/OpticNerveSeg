# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm3d(nn.Module):
    """LayerNorm for channel-last (B, D, H, W, C) or channel-first (B, C, D, H, W)"""
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5 and x.shape[-1] == self.weight.shape[0]:  # channel-last: (B, D, H, W, C)
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x * self.weight[None, None, None, None, :] + self.bias[None, None, None, None, :]
        else:  # channel-first: (B, C, D, H, W)
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class Block3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size, window_size),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            D, H, W = x.shape[1], x.shape[2], x.shape[3]
            x, pad_dhw = window_partition3D(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition3D(x, self.window_size, pad_dhw, (D, H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[2] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, D * H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, D * H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_d, self.rel_pos_h, self.rel_pos_w, (D, H, W), (D, H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, D, H, W, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, D, H, W, -1)
        x = self.proj(x)
        return x


def window_partition3D(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    B, D, H, W, C = x.shape
    pad_d = (window_size - D % window_size) % window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
    Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w
    x = x.view(B, Dp // window_size, window_size, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows, (Dp, Hp, Wp)


def window_unpartition3D(windows: torch.Tensor, window_size: int, pad_dhw: Tuple[int, int, int], dhw: Tuple[int, int, int]) -> torch.Tensor:
    Dp, Hp, Wp = pad_dhw
    D, H, W = dhw
    B = windows.shape[0] // ((Dp // window_size) * (Hp // window_size) * (Wp // window_size))
    x = windows.view(B, Dp // window_size, Hp // window_size, Wp // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Dp, Hp, Wp, -1)
    if Dp > D or Hp > H or Wp > W:
        x = x[:, :D, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1), size=max_rel_dist, mode="linear")
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn: torch.Tensor, q: torch.Tensor, rel_pos_d: torch.Tensor, rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor,
                           q_size: Tuple[int, int, int], k_size: Tuple[int, int, int]) -> torch.Tensor:
    q_d, q_h, q_w = q_size
    k_d, k_h, k_w = k_size
    Rd = get_rel_pos(q_d, k_d, rel_pos_d)
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_d, q_h, q_w, dim)
    rel_d = torch.einsum("bdhwc,dkc->bdhwk", r_q, Rd)
    rel_h = torch.einsum("bdhwc,hkc->bdhwk", r_q, Rh)
    rel_w = torch.einsum("bdhwc,wkc->bdhwk", r_q, Rw)
    attn = (attn.view(B, q_d, q_h, q_w, k_d, k_h, k_w) +
            rel_d[..., None, None] +
            rel_h[..., None, :, None] +
            rel_w[..., None, None, :]).view(B, q_d * q_h * q_w, k_d * k_h * k_w)
    return attn


class PatchEmbed3D(nn.Module):
    def __init__(self, kernel_size=(16,16,16), stride=(16,16,16), padding=(0,0,0), in_chans=1, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, C, D, H, W)
        x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        return x


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)


class ConvNext_Block3D(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm3d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) -> (B, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # back to (B, C, D, H, W)
        x = input + self.drop_path(x)
        return x


class ChannelAttentionModule3D(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv3d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = nn.Conv3d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        return self.sigmoid(self.conv3d(out))


class CBAM3D(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel_attention = ChannelAttentionModule3D(channel)
        self.spatial_attention = SpatialAttentionModule3D()

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class ConvNeXt3D(nn.Module):
    def __init__(self, in_chans=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=4, stride=4),
            Permute((0,2,3,4,1)),
            LayerNorm3d(dims[0]),
            Permute((0,4,1,2,3)),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                Permute((0,2,3,4,1)),
                LayerNorm3d(dims[i]),
                Permute((0,4,1,2,3)),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[ConvNext_Block3D(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        self.cbams = nn.ModuleList([CBAM3D(dims[i]) for i in range(4)])

    def forward_features(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = x + self.cbams[i](x)
            features.append(x)
            x = self.stages[i](x)
        return features


class FPN3D(nn.Module):
    def __init__(self, out_chans=256):
        super().__init__()
        dims = [768, 384, 192, 96]
        self.lat_layers = nn.ModuleList([nn.Conv3d(dims[i], out_chans, kernel_size=1) for i in range(4)])
        self.upconv_layers = nn.ModuleList([nn.ConvTranspose3d(out_chans, out_chans, kernel_size=2, stride=2) for _ in range(3)])
        self.high_res_layers = nn.ModuleList([nn.Sequential(
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_chans),
            nn.ReLU(inplace=True)
        ) for _ in range(3)])
        self.final_conv = nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, features):
        features = features[::-1]
        x = self.lat_layers[0](features[0])
        for i in range(1, 4):
            x = self.upconv_layers[i-1](x)
            x = x + self.lat_layers[i](features[i])
            x = self.high_res_layers[i-1](x)
            x = F.gelu(x)
        x = self.final_conv(x)
        return x


class CrossAttentionFusion3D(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out_conv = nn.Conv3d(d_model, d_model, kernel_size=1)

    def forward(self, conv_features, vit_features):
        B, C, D, H, W = conv_features.shape
        conv_flat = conv_features.flatten(2).permute(0, 2, 1)
        vit_flat = vit_features.flatten(2).permute(0, 2, 1)
        attn_output, _ = self.multihead_attn(conv_flat, vit_flat, vit_flat)
        attn_output = attn_output.permute(0, 2, 1).view(B, C, D, H, W)
        output = conv_features + attn_output
        output = output.permute(0, 2, 3, 4, 1)
        output = self.norm(output)
        output = output.permute(0, 4, 1, 2, 3)
        output = self.out_conv(output)
        return output


class ImageEncoderViT3D1(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed3D(
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, img_size // patch_size, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block3D(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            Permute((0,4,1,2,3)),
            nn.Conv3d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm3d(out_chans),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm3d(out_chans),
        )

        self.convnext = ConvNeXt3D(in_chans=in_chans)
        self.fpn = FPN3D(out_chans=out_chans)
        self.cross_fusion = CrossAttentionFusion3D(d_model=out_chans, n_heads=8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv_features = self.convnext.forward_features(x)
        conv_fpn_feature = self.fpn(conv_features)

        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        vit_feature = self.neck(x)  # (B, 256, 16, 16, 16)

        conv_fpn_feature = F.adaptive_avg_pool3d(conv_fpn_feature, vit_feature.shape[2:])

        fused_feature = self.cross_fusion(vit_feature, conv_fpn_feature)

        # Return fused (bottleneck), skip at 32^3 (conv_features[1]), skip at 64^3 (conv_features[0])
        return fused_feature, conv_features[1], conv_features[0]


# class UNetDecoder3D(nn.Module):
#     def __init__(self, bottleneck_chans=256, num_classes=1):
#         super().__init__()
#         # Decoder channels: start from bottleneck 256, down to 16
#         dec_chans = [128, 64, 32, 16]

#         # Up1: 16^3 -> 32^3, concat with skip2 (32^3 x 192) -> input chans 128 + 192
#         self.up1 = nn.ConvTranspose3d(bottleneck_chans, dec_chans[0], kernel_size=2, stride=2)
#         self.conv1 = nn.Sequential(
#             nn.Conv3d(dec_chans[0] + 192, dec_chans[0], kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(dec_chans[0], dec_chans[0], kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#         # Up2: 32^3 -> 64^3, concat with skip1 (64^3 x 96) -> 64 + 96
#         self.up2 = nn.ConvTranspose3d(dec_chans[0], dec_chans[1], kernel_size=2, stride=2)
#         self.conv2 = nn.Sequential(
#             nn.Conv3d(dec_chans[1] + 96, dec_chans[1], kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(dec_chans[1], dec_chans[1], kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#         # Up3: 64^3 -> 128^3, no skip, just conv
#         self.up3 = nn.ConvTranspose3d(dec_chans[1], dec_chans[2], kernel_size=2, stride=2)
#         self.conv3 = nn.Sequential(
#             nn.Conv3d(dec_chans[2], dec_chans[2], kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(dec_chans[2], dec_chans[2], kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#         # Up4: 128^3 -> 256^3, no skip
#         self.up4 = nn.ConvTranspose3d(dec_chans[2], dec_chans[3], kernel_size=2, stride=2)
#         self.conv4 = nn.Sequential(
#             nn.Conv3d(dec_chans[3], dec_chans[3], kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(dec_chans[3], dec_chans[3], kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#         # Final conv to num_classes
#         self.final_conv = nn.Conv3d(dec_chans[3], num_classes, kernel_size=1)

#     def forward(self, bottleneck: torch.Tensor, skip2: torch.Tensor, skip1: torch.Tensor) -> torch.Tensor:
#         # bottleneck: (B, 256, 16, 16, 16)
#         # skip2: (B, 192, 32, 32, 32)
#         # skip1: (B, 96, 64, 64, 64)

#         x = self.up1(bottleneck)  # (B, 128, 32, 32, 32)
#         x = torch.cat([x, skip2], dim=1)  # (B, 128+192, 32, 32, 32)
#         x = self.conv1(x)  # (B, 128, 32, 32, 32)

#         x = self.up2(x)  # (B, 64, 64, 64, 64)
#         x = torch.cat([x, skip1], dim=1)  # (B, 64+96, 64, 64, 64)
#         x = self.conv2(x)  # (B, 64, 64, 64, 64)

#         x = self.up3(x)  # (B, 32, 128, 128, 128)
#         x = self.conv3(x)  # (B, 32, 128, 128, 128)

#         x = self.up4(x)  # (B, 16, 256, 256, 256)
#         x = self.conv4(x)  # (B, 16, 256, 256, 256)

#         x = self.final_conv(x)  # (B, num_classes, 256, 256, 256)

#         return x


# class FullModel3D(nn.Module):
#     def __init__(
#         self,
#         img_size: int = 256,
#         patch_size: int = 16,
#         in_chans: int = 1,
#         num_classes: int = 2,
#         embed_dim: int = 768,
#         depth: int = 12,
#         num_heads: int = 12,
#         mlp_ratio: float = 4.0,
#         out_chans: int = 256,
#         qkv_bias: bool = True,
#         norm_layer: Type[nn.Module] = nn.LayerNorm,
#         act_layer: Type[nn.Module] = nn.GELU,
#         use_abs_pos: bool = True,
#         use_rel_pos: bool = False,
#         rel_pos_zero_init: bool = True,
#         window_size: int = 0,
#         global_attn_indexes: Tuple[int, ...] = (),
#     ) -> None:
#         super().__init__()
#         self.encoder = ImageEncoderViT3D1(
#             img_size=img_size,
#             patch_size=patch_size,
#             in_chans=in_chans,
#             embed_dim=embed_dim,
#             depth=depth,
#             num_heads=num_heads,
#             mlp_ratio=mlp_ratio,
#             out_chans=out_chans,
#             qkv_bias=qkv_bias,
#             norm_layer=norm_layer,
#             act_layer=act_layer,
#             use_abs_pos=use_abs_pos,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             window_size=window_size,
#             global_attn_indexes=global_attn_indexes,
#         )
#         self.decoder = UNetDecoder3D(bottleneck_chans=out_chans, num_classes=num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         bottleneck, skip2, skip1 = self.encoder(x)
#         output = self.decoder(bottleneck, skip2, skip1)
#         return output

class DualEncoderUNet3D(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 1,
        num_classes: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()

        # Two separate encoders (you can also share weights if desired)
        self.encoder1 = ImageEncoderViT3D1(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_chans=out_chans,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_abs_pos=use_abs_pos,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
        )

        self.encoder2 = ImageEncoderViT3D1(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_chans=out_chans,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_abs_pos=use_abs_pos,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
        )

        # Modified decoder for doubled channels
        self.decoder = UNetDecoder3D_Dual(
            bottleneck_chans=out_chans * 2,   # 256 -> 512 after cat
            skip2_chans=192 * 2,             # 192 -> 384
            skip1_chans=96 * 2,              # 96 -> 192
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward through both encoders
        bottleneck1, skip2_1, skip1_1 = self.encoder1(x)
        bottleneck2, skip2_2, skip1_2 = self.encoder2(x)

        # Fuse by concatenation along channel dimension
        fused_bottleneck = torch.cat([bottleneck1, bottleneck2], dim=1)  # (B, 512, 16, 16, 16)
        fused_skip2 = torch.cat([skip2_1, skip2_2], dim=1)              # (B, 384, 32, 32, 32)
        fused_skip1 = torch.cat([skip1_1, skip1_2], dim=1)              # (B, 192, 64, 64, 64)

        # Decode
        output = self.decoder(fused_bottleneck, fused_skip2, fused_skip1)
        return output


class UNetDecoder3D_Dual(nn.Module):
    def __init__(self, bottleneck_chans=512, skip2_chans=384, skip1_chans=192, num_classes=1):
        super().__init__()
        dec_chans = [256, 128, 64, 32]  # Adjustable internal decoder channels

        # Up1: 16→32, concat with skip2 (384) → input: 256 + 384 = 640
        self.up1 = nn.ConvTranspose3d(bottleneck_chans, dec_chans[0], kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv3d(dec_chans[0] + skip2_chans, dec_chans[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dec_chans[0], dec_chans[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Up2: 32→64, concat with skip1 (192) → input: 128 + 192 = 320
        self.up2 = nn.ConvTranspose3d(dec_chans[0], dec_chans[1], kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv3d(dec_chans[1] + skip1_chans, dec_chans[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dec_chans[1], dec_chans[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Up3: 64→128
        self.up3 = nn.ConvTranspose3d(dec_chans[1], dec_chans[2], kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv3d(dec_chans[2], dec_chans[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dec_chans[2], dec_chans[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Up4: 128→256
        self.up4 = nn.ConvTranspose3d(dec_chans[2], dec_chans[3], kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv3d(dec_chans[3], dec_chans[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dec_chans[3], dec_chans[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv3d(dec_chans[3], num_classes, kernel_size=1)

    def forward(self, bottleneck: torch.Tensor, skip2: torch.Tensor, skip1: torch.Tensor) -> torch.Tensor:
        x = self.up1(bottleneck)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = self.conv3(x)

        x = self.up4(x)
        x = self.conv4(x)

        x = self.final_conv(x)
        return x

# if __name__ == "__main__":
#     model = DualEncoderUNet3D(img_size=128, num_classes=2)
#     input_tensor = torch.randn(1, 1, 128, 128, 128)
    
#     output = model(input_tensor)
#     print("Final output shape:", output.shape)  # Should be [1, 2, 128, 128, 128]
# # if __name__ == "__main__":
# #     model = FullModel3D(img_size=128, num_classes=1)
# #     input_tensor = torch.randn(1, 1, 128, 128, 128)
# #     output = model(input_tensor)
# #     print("Output shape:", output.shape)  # Should be [1, 1, 128, 128, 128]