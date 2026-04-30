import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .hiet_layer import HiETLayer

class HiETBlock(nn.Module):
    def __init__(self, dim: int = 96, num_heads: int = 8, depth: int = 3,
                 mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.encoder_window_sizes = [(64, 64), (32, 32), (8, 8)]
        self.decoder_window_sizes = [(8, 8), (32, 32), (64, 64)]

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        
        for i in range(depth):
            # Encoder layers
            self.encoder_layers.append(
                HiETLayer(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.encoder_window_sizes[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop
                )
            )
            # Decoder layers
            self.decoder_layers.append(
                HiETLayer(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.decoder_window_sizes[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop
                )
            )

        self.bottleneck = HiETLayer(
            dim=dim,
            num_heads=num_heads,
            window_size=(4, 4),
            mlp_ratio=mlp_ratio,
            drop=drop
        )

        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        for i in range(depth):
            downsample = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
                # Custom LayerNorm for [B, C, H, W] input
                nn.GroupNorm(1, dim) # Equivalent to LayerNorm over the channel dimension
            )
            self.downsample_layers.append(downsample)

            upsample = nn.Sequential(
                nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(1, dim)
            )
            self.upsample_layers.append(upsample)

        self.skip_fusion = nn.ModuleList()
        for i in range(depth):
            fusion = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )
            self.skip_fusion.append(fusion)

        self.input_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)

        self.norm_in = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) # B, H, W, C

        x = self.norm_in(x)
        x = self.input_proj(x)

        encoder_features = []
        current_x = x

        # Encoder path
        for i in range(self.depth):
            current_x = self.encoder_layers[i](current_x)
            encoder_features.append(current_x)

            if i < self.depth - 1:
                current_x_conv = current_x.permute(0, 3, 1, 2)
                current_x_conv = self.downsample_layers[i](current_x_conv)
                current_x = current_x_conv.permute(0, 2, 3, 1)

        # Bottleneck
        current_x = self.bottleneck(current_x)

        # Decoder path
        for i in range(self.depth - 1, -1, -1):
            current_x_conv = current_x.permute(0, 3, 1, 2)
            current_x_conv = self.upsample_layers[i](current_x_conv)
            current_x = current_x_conv.permute(0, 2, 3, 1)

            skip_feature = encoder_features[i]

            if current_x.shape[1] != skip_feature.shape[1] or current_x.shape[2] != skip_feature.shape[2]:
                current_x = F.interpolate(
                    current_x.permute(0, 3, 1, 2),
                    size=(skip_feature.shape[1], skip_feature.shape[2]),
                    mode='bilinear',
                    align_corners=False,
                ).permute(0, 2, 3, 1)

            fused = torch.cat([current_x, skip_feature], dim=-1)
            current_x = self.skip_fusion[i](fused)
            current_x = self.decoder_layers[i](current_x)

        x = self.output_proj(current_x)
        x = self.norm_out(x)
        x = x.permute(0, 3, 1, 2) # B, C, H, W

        return x

class MultiScaleHiETBlock(nn.Module):
    def __init__(self, dim: int = 96, num_heads: int = 8, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.scale_blocks = nn.ModuleList()

        for i in range(num_scales):
            depth = 3 - i
            block = HiETBlock(
                dim=dim, 
                num_heads=num_heads,
                depth=depth
            )
            self.scale_blocks.append(block)

        self.scale_fusion = nn.Sequential(
            nn.Linear(dim * num_scales, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_outputs = []

        for i, block in enumerate(self.scale_blocks):
            if i > 0:
                scale_factor = 2 ** i
                x_scaled = F.interpolate(
                    x,
                    scale_factor=1/scale_factor,
                    mode='bilinear',
                    align_corners=False
                )
                out = block(x_scaled)
                out = F.interpolate(
                    out,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                out = block(x)

            scale_outputs.append(out)

        fused = torch.cat(scale_outputs, dim=1)
        fused = fused.permute(0, 2, 3, 1)
        fused = self.scale_fusion(fused)
        fused = fused.permute(0, 3, 1, 2)

        return fused
