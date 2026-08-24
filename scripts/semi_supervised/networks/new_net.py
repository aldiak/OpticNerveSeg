import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2), DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad to match spatial dimensions
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]
        x1 = F.pad(
            x1,
            [
                diffW // 2,
                diffW - diffW // 2,
                diffH // 2,
                diffH - diffH // 2,
                diffD // 2,
                diffD - diffD // 2,
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetEncoder3D(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        self.inc = DoubleConv3D(in_ch, base_ch)
        self.down1 = Down3D(base_ch, base_ch * 2)
        self.down2 = Down3D(base_ch * 2, base_ch * 4)
        self.down3 = Down3D(base_ch * 4, base_ch * 8)
        self.down4 = Down3D(base_ch * 8, base_ch * 16)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]


class FPN3D(nn.Module):
    """Simple 3D Feature Pyramid Network (top-down pathway with lateral connections)"""

    def __init__(self, in_channels_list=[64, 128, 256, 512, 1024], out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv3d(in_ch, out_channels, kernel_size=1)
                for in_ch in in_channels_list
            ]
        )
        self.smooth_convs = nn.ModuleList(
            [
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in in_channels_list
            ]
        )

    def forward(self, features):
        # features: [level0 (high-res), level1, ..., level4 (low-res)]
        laterals = [
            lateral(feat) for lateral, feat in zip(self.lateral_convs, features)
        ]

        # Start from lowest resolution
        out = laterals[-1]
        fpn_features = [out]
        for i in reversed(range(len(laterals) - 1)):
            up = F.interpolate(
                out, size=laterals[i].shape[2:], mode="trilinear", align_corners=True
            )
            out = laterals[i] + up
            out = self.smooth_convs[i](out)
            fpn_features.append(out)

        return fpn_features[::-1]  # Return high-res first: [p0, p1, p2, p3, p4]


class UNetDecoderPlusWithFPNAndUncertainty3D(nn.Module):
    """Decoder with FPN integration and uncertainty estimation branch"""

    def __init__(self, out_seg_ch=2, base_ch=64, fpn_channels=256, bilinear=False):
        super().__init__()
        self.bilinear = bilinear

        # Upsampling layers (now include FPN channels in input)
        self.up1 = Up3D(base_ch * 16 + fpn_channels, base_ch * 8, bilinear)
        self.up2 = Up3D(base_ch * 8 + fpn_channels, base_ch * 4, bilinear)
        self.up3 = Up3D(base_ch * 4 + fpn_channels, base_ch * 2, bilinear)
        self.up4 = Up3D(base_ch * 2 + fpn_channels, base_ch, bilinear)

        # Final outputs
        self.out_seg = OutConv3D(base_ch, out_seg_ch)
        self.out_uncert = OutConv3D(
            base_ch, 1
        )  # Predicted variance (aleatoric uncertainty)

    def forward(self, fused_features, encoder_features):
        # fused_features: list of 5 fused cross-branch features
        # encoder_features: list of 5 own-branch encoder features
        fpn_feats = FPN3D()(fused_features)  # [high-res to low-res]

        # Bottleneck + FPN low-res
        x = torch.cat([encoder_features[4], fpn_feats[4]], dim=1)
        x = self.up1(x, encoder_features[3])
        x = torch.cat([x, fpn_feats[3]], dim=1)

        x = self.up2(x, encoder_features[2])
        x = torch.cat([x, fpn_feats[2]], dim=1)

        x = self.up3(x, encoder_features[1])
        x = torch.cat([x, fpn_feats[1]], dim=1)

        x = self.up4(x, encoder_features[0])
        x = torch.cat([x, fpn_feats[0]], dim=1)

        seg_logits = self.out_seg(x)
        uncert = F.softplus(self.out_uncert(x))  # Ensure positive variance

        return seg_logits, uncert


class OpticNerveNet3D_Enhanced(nn.Module):
    """
    Enhanced 3D network for optic nerve segmentation with:
    - Innovation 3: Feature Pyramid Network (FPN) for multi-scale context
    - Innovation 6: Uncertainty estimation branch (aleatoric uncertainty)
    - (Innovation 4: DTI alignment loss provided separately below)
    """

    def __init__(self, in_chs=[7, 3, 3, 4], out_seg_ch=2, base_ch=64, bilinear=False):
        super().__init__()

        # Initial coarse segmentation UNet (full 3D UNet for combined modalities)
        self.unet_initial = nn.Sequential(
            DoubleConv3D(in_chs[0], base_ch),
            Down3D(base_ch, base_ch * 2),
            Down3D(base_ch * 2, base_ch * 4),
            Down3D(base_ch * 4, base_ch * 8),
            Down3D(base_ch * 8, base_ch * 16),
            Up3D(base_ch * 16 + base_ch * 8, base_ch * 8, bilinear),
            Up3D(base_ch * 8 + base_ch * 4, base_ch * 4, bilinear),
            Up3D(base_ch * 4 + base_ch * 2, base_ch * 2, bilinear),
            Up3D(base_ch * 2 + base_ch, base_ch, bilinear),
            OutConv3D(base_ch, out_seg_ch),
        )

        # Branch encoders
        self.encoder_struct = UNetEncoder3D(in_ch=in_chs[1], base_ch=base_ch)
        self.encoder_dti_aniso = UNetEncoder3D(in_ch=in_chs[2], base_ch=base_ch)
        self.encoder_dti_dir = UNetEncoder3D(in_ch=in_chs[3], base_ch=base_ch)

        # Enhanced decoders with FPN and uncertainty
        self.decoder_struct = UNetDecoderPlusWithFPNAndUncertainty3D(
            out_seg_ch=out_seg_ch, base_ch=base_ch
        )
        self.decoder_dti_aniso = UNetDecoderPlusWithFPNAndUncertainty3D(
            out_seg_ch=out_seg_ch, base_ch=base_ch
        )
        self.decoder_dti_dir = UNetDecoderPlusWithFPNAndUncertainty3D(
            out_seg_ch=out_seg_ch, base_ch=base_ch
        )

    def forward(self, T1, T2, FA, MD, Eig):
        # Combined input for initial segmentation
        # img_combined = torch.cat([T1, T2, FA, MD, Eig], dim=1)
        img_combined = torch.cat([T1, FA], dim=1)
        seg_initial = self.unet_initial(img_combined)
        mask_initial = torch.argmax(seg_initial, dim=1, keepdim=True).float()

        # Branch inputs with initial mask
        img_struct = torch.cat([T1, T2, mask_initial], dim=1)
        img_dti_aniso = torch.cat([FA, MD, mask_initial], dim=1)
        img_dti_dir = torch.cat([Eig, mask_initial], dim=1)

        # Encode
        f_struct = self.encoder_struct(img_struct)
        f_dti_aniso = self.encoder_dti_aniso(img_dti_aniso)
        f_dti_dir = self.encoder_dti_dir(img_dti_dir)

        # Cross-branch fusion (max across the other two branches)
        fused_struct = [torch.max(f_dti_aniso[i], f_dti_dir[i]) for i in range(5)]
        fused_aniso = [torch.max(f_struct[i], f_dti_dir[i]) for i in range(5)]
        fused_dir = [torch.max(f_struct[i], f_dti_aniso[i]) for i in range(5)]

        # Decode with FPN + uncertainty
        seg_struct, uncert_struct = self.decoder_struct(fused_struct, f_struct)
        seg_aniso, uncert_aniso = self.decoder_dti_aniso(fused_aniso, f_dti_aniso)
        seg_dir, uncert_dir = self.decoder_dti_dir(fused_dir, f_dti_dir)

        # Final segmentation: ensemble average of the three refined outputs
        final_seg = (seg_struct + seg_aniso + seg_dir) / 3.0

        return (
            seg_initial,
            final_seg,
            (seg_struct, seg_aniso, seg_dir),
            (uncert_struct, uncert_aniso, uncert_dir),
        )

        return (
            seg_initial,
            final_seg,
            (seg_struct, seg_aniso, seg_dir),
            (uncert_struct, uncert_aniso, uncert_dir),
        )


# Example usage and shapes (tested with dummy data)
if __name__ == "__main__":
    model = OpticNerveNet3D_Enhanced()
    B, D, H, W = 1, 64, 128, 128
    T1 = torch.randn(B, 1, D, H, W)
    T2 = torch.randn(B, 1, D, H, W)
    FA = torch.randn(B, 1, D, H, W)
    MD = torch.randn(B, 1, D, H, W)
    Eig = torch.randn(B, 3, D, H, W)

    seg_initial, final_seg, branch_segs, uncertainties = model(T1, T2, FA, MD, Eig)
    print("seg_initial shape:", seg_initial.shape)
    print("final_seg shape:", final_seg.shape)
    print("branch segs shapes:", [s.shape for s in branch_segs])
    print("uncertainty maps shapes:", [u.shape for u in uncertainties])
