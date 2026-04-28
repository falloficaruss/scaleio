import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class HierarchicalCoordinateEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(
        self, h: int, w: int, scale: float, device: torch.device
    ) -> torch.Tensor:
        y_coords = torch.arange(h, device=device).float()
        x_coords = torch.arange(w, device=device).float()

        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        yy_norm = yy / (h - 1) if h > 1 else yy
        xx_norm = xx / (w - 1) if w > 1 else xx

        cell_size = torch.tensor([2.0 / scale, 2.0 / scale], device=device)
        coords = torch.stack([yy_norm.flatten(), xx_norm.flatten()], dim=-1) # (H*W, 2)
        encoding = []

        for i in range(self.d_model // 8):
            freq = 2.0**i

            cell_coords = coords * cell_size.unsqueeze(0)
            encoding.append(torch.sin(cell_coords[:, 0] * freq * math.pi))
            encoding.append(torch.cos(cell_coords[:, 0] * freq * math.pi))
            encoding.append(torch.sin(cell_coords[:, 1] * freq * math.pi))
            encoding.append(torch.cos(cell_coords[:, 1] * freq * math.pi))

            encoding.append(torch.sin(coords[:, 0] * freq * math.pi))
            encoding.append(torch.cos(coords[:, 0] * freq * math.pi))
            encoding.append(torch.sin(coords[:, 1] * freq * math.pi))
            encoding.append(torch.cos(coords[:, 1] * freq * math.pi))

        remaining = self.d_model - len(encoding)

        if remaining > 0:
            for _ in range(remaining):
                encoding.append(torch.zeros_like(coords[:, 0]))

        coord_encoding = torch.stack(encoding, dim=-1)
        return coord_encoding


class LinearAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.0)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = F.softmax(q, dim=-1) * self.scale
        k = F.softmax(k, dim=-2)

        # Linear attention: (Q @ (K.T @ V))
        # k: B, H, N, D; v: B, H, N, D
        # context: B, H, D, D
        context = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, context) # B, H, N, D

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class HIIFL(nn.Module):
    def __init__(self, dim: int = 96, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim

        self.coord_encoder = HierarchicalCoordinateEncoding(dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(dim * 2, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        self.linear_attn = LinearAttention(dim, num_heads)

        self.mlp2 = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(int(dim * mlp_ratio), 3),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor, scale: float) -> torch.Tensor:
        B, C, H_lr, W_lr = features.shape

        H_hr = int(H_lr * scale)
        W_hr = int(W_lr * scale)

        coord_encoding = self.coord_encoder(H_hr, W_hr, scale, features.device)
        coord_encoding = coord_encoding.unsqueeze(0).expand(B, -1, -1)

        features_up = F.interpolate(
            features, size=(H_hr, W_hr), mode="bilinear", align_corners=False
        )

        features_flat = features_up.permute(0, 2, 3, 1).reshape(B, H_hr * W_hr, C)

        combined = torch.cat([features_flat, coord_encoding], dim=-1)

        x = self.mlp1(combined)
        x = self.norm1(x)

        x_attn = self.linear_attn(x)
        x = x + x_attn
        x = self.norm2(x)

        rgb_flat = self.mlp2(x)

        rgb_output = rgb_flat.reshape(B, H_hr, W_hr, 3).permute(0, 3, 1, 2)

        rgb_output = torch.tanh(rgb_output) * 0.5 + 0.5
        rgb_output = rgb_output.clamp(0.0, 1.0)

        return rgb_output


class AdaptiveHIIFL(nn.Module):
    def __init__(self, input_dim: int = 96, dim: int = 96, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim

        if input_dim != dim:
            self.input_proj = nn.Linear(input_dim, dim)
        else:
            self.input_proj = nn.Identity()

        self.hiif_l = HIIFL(dim, num_heads)

    def forward(self, features: torch.Tensor, scale: float) -> torch.Tensor:
        if hasattr(self, "input_proj") and not isinstance(self.input_proj, nn.Identity):
            B, C, H, W = features.shape
            features_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, C)
            features_proj = self.input_proj(features_flat)
            features = features_proj.reshape(B, H, W, self.dim).permute(0, 3, 1, 2)

        return self.hiif_l(features, scale)
