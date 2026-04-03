import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class HierarchicalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        y_coords = torch.arange(h, device=device).float()
        x_coords = torch.arange(h, device=device).float()

        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        yy = yy / (h - 1) if h > 1 else yy
        xx = xx / (w - 1) if w > 1 else xx

        coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)

        encoding = []
        for i in range(self.d_model // 4):
            freq = 2.0**i
            encoding.append(torch.sin(coords[:, 0] * freq * math.pi))
            encoding.append(torch.cos(coords[:, 0] * freq * math.pi))
            encoding.append(torch.sin(coords[:, 1] * freq * math.pi))
            encoding.append(torch.cos(coords[:, 1] * freq * math.pi))

        if self.d_model % 4 != 0:
            encoding.append(torch.zeros_like(coords[:, 0]))

        delta_hw = torch.stack(encoding, dim=-1)
