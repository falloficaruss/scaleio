import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
from .hiet_block import HiETBlock
from .hiif_l import HIIFL, AdaptiveHIIFL


class ShallowFeatureExtractor(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 96):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class SubPixelUpsampler(nn.Module):

    def __init__(self, in_channels: int = 96, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor * scale_factor,
            kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class C2DISR(nn.Module):

    def __init__(self,
        in_channels: int = 3,
        feature_dim: int = 96,
        num_hiet_blocks: int = 3,
        num_heads: int = 8,
        stage: str = 'continuous',
        scale_factor: Optional[int] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.num_hiet_blocks = num_hiet_blocks
        self.num_heads = num_heads
        self.stage = stage
        self.scale_factor = scale_factor

        assert stage in ['continuous', 'discrete'], f"Invalid stage: {stage}"
        if stage == 'discrete':
            assert scale_factor is not None, "scale_factor required for discrete stage"
            assert scale_factor in [2, 3, 4], f"Unsupported scale factor: {scale_factor}"

        self.shallow_extractor = ShallowFeatureExtractor(in_channels, feature_dim)

        self.deep_extractor = HiETBlock(
            dim=feature_dim,
            num_heads=num_heads,
            depth=num_hiet_blocks
        )

        if stage == 'continuous':
            self.upsampler = HIIFL(
                dim=feature_dim,
                num_heads=num_heads
            )

        else:
            self.upsampler = SubPixelUpsampler(
                in_channels=feature_dim,
                scale_factor=scale_factor
            )

            self.rgb_proj = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim // 2, 3, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
        shallow_features = self.shallow_extractor(x)
        deep_features = self.deep_extractor(shallow_features)

        if self.stage == 'continuous':
            assert scale is not None, "Scale factor required for continuous stage"
            sr_image = self.upsampler(deep_features, scale)
        else:
            upsampled_features = self.upsampler(deep_features)
            sr_image = self.rgb_proj(upsampled_features)

        return sr_image

    def get_features(self, x: torch.Tensor) -> tuple:
        shallow_features = self.shallow_extractor(x)
        deep_features = self.deep_extractor(shallow_features)

        return shallow_features, deep_features

    def load_stage1_weights(self, stage1_checkpoint: str, strict: bool = True):
        checkpoint = torch.load(stage1_checkpoint, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        filtered_state_dict = {}

        for key, value in state_dict.items():
            if not key.startswith('upsampler'):
                filtered_state_dict[key] = value

        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=strict)

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        print(f"Loaded Stage 1 weights, filtered out upsampler for Stage 2 fine-tuning")


class C2DISRFactory:

    @staticmethod
    def create_stage1_model(feature_dim: int = 96, num_hiet_blocks: int = 3,
        num_heads: int = 8) -> C2DISR:
        return C2DISR(
            in_channels=3,
            feature_dim=feature_dim,
            num_hiet_blocks=num_hiet_blocks,
            num_heads=num_heads,
            stage='continuous'
        )

    @staticmethod
    def create_model_from_stage1(stage1_model: C2DISR, scale_factor: int = 4) -> C2DISR:
        return C2DISR(
            in_channels=stage1_model.in_channels,
            feature_dim=stage1_model.feature_dim,
            num_hiet_blocks=stage1_model.num_hiet_blocks,
            num_heads=stage1_model.num_heads,
            stage='discrete',
            scale_factor=scale_factor
        )

    @staticmethod
    def count_parameters(model: nn.Module) -> tuple:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return total_params, trainable_params

    @staticmethod
    def get_model_info(model: C2DISR) -> dict:
        total_params, trainable_params = C2DISRFactory.count_parameters(model)

        return {
            'stage': model.stage,
            'feature_dim': model.feature_dim,
            'num_hiet_blocks': model.num_hiet_blocks,
            'scale_factor': model.scale_factor,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameters_mb': total_params / (1024 * 1024)
        }
