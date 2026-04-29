import random
from pathlib import Path

import kornia
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SRDataset(Dataset):
    def __init__(self, hr_dir, scale_factor=4, patch_size=192, augment=True):
        self.hr_dir = Path(hr_dir)
        self.hr_paths = list(self.hr_dir.glob("*.png")) + list(
            self.hr_dir.glob("*.jpg")
        )
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]
        hr_img = Image.open(hr_path).convert("RGB")
        hr_img_np = np.array(hr_img)
        hr_tensor = kornia.image_to_tensor(hr_img_np, keepdim=False).float() / 255.0
        hr_tensor = hr_tensor.squeeze(0)

        h, w = hr_tensor.shape[-2:]
        if h > self.patch_size or w > self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            hr_tensor = hr_tensor[
                :, top : top + self.patch_size, left : left + self.patch_size
            ]

        lr_tensor = kornia.geometry.transform.resize(
            hr_tensor,
            (
                self.patch_size // self.scale_factor,
                self.patch_size // self.scale_factor,
            ),
            interpolation="bicubic",
        )

        if self.augment and random.random() > 0.5:
            hr_tensor = torch.flip(hr_tensor, dims=[2])
            lr_tensor = torch.flip(lr_tensor, dims=[2])
        if self.augment and random.random() > 0.5:
            hr_tensor = torch.flip(hr_tensor, dims=[1])
            lr_tensor = torch.flip(lr_tensor, dims=[1])

        return lr_tensor, hr_tensor


class ContinuousScaleData(Dataset):
    def __init__(
        self, hr_dir, min_scale=1.0, max_scale=4.0, patch_size=192, augment=True
    ):
        self.hr_dir = Path(hr_dir)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.patch_size = patch_size
        self.augment = augment
        self.hr_paths = list(self.hr_dir.glob("*.png")) + list(
            self.hr_dir.glob("*.jpg")
        )

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]
        hr_img = Image.open(hr_path).convert("RGB")
        hr_img_np = np.array(hr_img)
        hr_tensor = kornia.image_to_tensor(hr_img_np, keepdim=False).float() / 255.0
        hr_tensor = hr_tensor.squeeze(0)

        h, w = hr_tensor.shape[-2:]
        if h > self.patch_size or w > self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            hr_tensor = hr_tensor[
                :, top : top + self.patch_size, left : left + self.patch_size
            ]

        if self.augment and random.random() > 0.5:
            hr_tensor = torch.flip(hr_tensor, dims=[2])
        if self.augment and random.random() > 0.5:
            hr_tensor = torch.flip(hr_tensor, dims=[1])

        return hr_tensor
