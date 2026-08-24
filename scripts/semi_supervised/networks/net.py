import torch
import torch.nn as nn
from typing import Sequence
import inspect
from monai.networks.nets import SwinUNETR as MONAISwinUNETR
from monai.networks.blocks import UnetrUpBlock

# Fixed MoEConvBlock - NO residual if channels differ
class MoEConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=3, normalization='batchnorm', num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.out_channels = out_channels
        
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            layers = []
            for i in range(n_convs):
                cin = in_channels if i == 0 else out_channels
                layers.append(nn.Conv3d(cin, out_channels, kernel_size=3, padding=1))
                if normalization == 'batchnorm':
                    layers.append(nn.BatchNorm3d(out_channels))
                elif normalization == 'groupnorm':
                    layers.append(nn.GroupNorm(16, out_channels))
                elif normalization == 'instancenorm':
                    layers.append(nn.InstanceNorm3d(out_channels))
                if i < n_convs - 1:
                    layers.append(nn.ReLU(inplace=True))
            self.experts.append(nn.Sequential(*layers))
        
        self.gate = nn.Sequential(
            nn.Conv3d(in_channels + num_experts, num_experts, kernel_size=1),
            nn.Sigmoid()
        )
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x, mod_prior):
        B, C, *spatial = x.shape
        if mod_prior.dim() == 2:
            mod_prior = mod_prior.view(B, -1, *[1]*len(spatial))
        mod_prior = mod_prior.expand(-1, -1, *spatial)
        
        gate_input = torch.cat([x, mod_prior], dim=1)
        gates = self.gate(gate_input)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gates = gates.unsqueeze(2)
        out = (expert_outputs * gates).sum(dim=1)
        
        # Only add residual if channels match
        if out.shape[1] == x.shape[1]:
            out = out + x
        return self.final_relu(out)

# Fixed MoEUnetrUpBlock
class MoEUnetrUpBlock(UnetrUpBlock):
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size, upsample_kernel_size, norm_name, num_experts=4, res_block=False):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=norm_name,
            res_block=res_block
        )
        self.num_experts = num_experts
        concat_channels = 2 * out_channels  # up + skip
        self.conv_block = MoEConvBlock(concat_channels, out_channels, n_convs=3, num_experts=num_experts)
        if res_block:
            self.conv_block2 = MoEConvBlock(out_channels, out_channels, n_convs=3, num_experts=num_experts)

    def forward(self, inp, skip, mod_prior):
        up = self.transp_conv(inp)
        out = torch.cat([up, skip], dim=1)
        out = self.conv_block(out, mod_prior)
        if hasattr(self, 'conv_block2'):
            out = self.conv_block2(out, mod_prior)
        return out

# Final MoESwinUNETR
class MoESwinUNETR(MONAISwinUNETR):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        img_size=(128, 128, 128),
        feature_size: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        norm_name: str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        num_experts: int = 4,
    ):
        parent_sig = inspect.signature(MONAISwinUNETR.__init__)
        init_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "feature_size": feature_size,
            "depths": depths,
            "num_heads": num_heads,
            "norm_name": norm_name,
            "drop_rate": drop_rate,
            "attn_drop_rate": attn_drop_rate,
            "dropout_path_rate": dropout_path_rate,
            "use_checkpoint": use_checkpoint,
            "spatial_dims": spatial_dims,
        }
        if "img_size" in parent_sig.parameters:
            init_kwargs["img_size"] = img_size
        super().__init__(**init_kwargs)

        self.num_experts = num_experts

        self.decoder5 = MoEUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            num_experts=num_experts,
        )
        self.decoder4 = MoEUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            num_experts=num_experts,
        )
        self.decoder3 = MoEUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            num_experts=num_experts,
        )
        self.decoder2 = MoEUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            num_experts=num_experts,
        )
        self.decoder1 = MoEUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            num_experts=num_experts,
        )

    def forward(self, x_in: torch.Tensor, mod_prior: torch.Tensor | None = None):
        if mod_prior is None:
            mod_prior = torch.ones(x_in.shape[0], self.num_experts, device=x_in.device) / self.num_experts

        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        dec3 = self.decoder5(dec4, hidden_states_out[3], mod_prior)
        dec2 = self.decoder4(dec3, hidden_states_out[2], mod_prior)
        dec1 = self.decoder3(dec2, hidden_states_out[1], mod_prior)
        dec0 = self.decoder2(dec1, hidden_states_out[0], mod_prior)
        out = self.decoder1(dec0, enc0, mod_prior)
        logits = self.out(out)
        return logits

# Test
# if __name__ == "__main__":
#     model = MoESwinUNETR(
#         in_channels=4,
#         out_channels=4,
#         img_size=(128, 128, 128),
#         feature_size=48,
#         depths=(2, 2, 2, 2),
#         num_heads=(3, 6, 12, 24),
#         spatial_dims=3,
#         num_experts=4
#     )

#     x = torch.randn(2, 4, 64, 64, 64)  # Batch of 2, 4 channels, 64x64x64 volume
#     prior = torch.tensor([[1.0, 0.0, 1.0, 1.0],
#                           [1.0, 0.0, 1.0, 1.0]])
#     out = model(x, prior)
#     print("Output shape:", out.shape)  # (2, 4, 128, 128, 128)