import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class L1Loss(nn.Module):
    """L1 loss (Mean Absolute Error) for super-resolution."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate L1 loss between prediction and target.

        Args:
            pred: predicted image tensor (B, C, H, W) in [0, 1]
            target: target image tensor (B, C, H, W) in [0, 1]

        Returns:
            L1 loss value
        """
        loss = F.l1_loss(pred, target, reduction=self.reduction)
        return loss

class CharbonnierLoss(nn.Module):
    """Charbonnier loss - robust L1 alternative."""

    def __init__(self, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Charbonnier loss.

        Args:
            pred: predicted image tensor (B, C, H, W) in [0, 1]
            target: target image tensor (B, C, H, W) in [0, 1]

        Returns:
            Charbonnier loss value
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class MSELoss(nn.Module):
    """MSE loss (Mean Squared Error) for super-resolution."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate MSE loss between prediction and target.

        Args:
            pred: predicted image tensor (B, C, H, W) in [0, 1]
            target: target image tensor (B, C, H, W) in [0, 1]

        Returns:
            MSE loss value
        """
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return loss

class CombinedLoss(nn.Module):
    """Combined loss with multiple loss functions."""

    def __init__(self, loss_configs: list):
        """
        Initialize combined loss.

        Args:
            loss_configs: list of dicts with 'type' and 'weight' keys
                         e.g., [{'type': 'l1', 'weight': 1.0}, {'type': 'mse', 'weight': 0.1}]
        """
        super().__init__()
        self.losses = nn.ModuleList()
        self.weights = []

        for config in loss_configs:
            loss_type = config['type'].lower()
            weight = config['weight']

            if loss_type == 'l1':
                loss = L1Loss()
            elif loss_type == 'mse':
                loss = MSELoss()
            elif loss_type == 'charbonnier':
                loss = CharbonnierLoss()
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")

            self.losses.append(loss)
            self.weights.append(weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.

        Args:
            pred: predicted image tensor (B, C, H, W) in [0, 1]
            target: target image tensor (B, C, H, W) in [0, 1]

        Returns:
            Combined loss value
        """
        total_loss = 0.0

        for loss, weight in zip(self.losses, self.weights):
            loss_value = loss(pred, target)
            total_loss += weight * loss_value

        return total_loss

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features (optional for future extensions)."""

    def __init__(self, feature_layers: list = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super().__init__()
        self.feature_layers = feature_layers

        # Load pretrained VGG16
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        self.features = vgg.features

        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False

        # Layer mapping
        self.layer_name_mapping = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3'
        }

    def extract_features(self, x: torch.Tensor) -> dict:
        """Extract VGG features."""
        features = {}
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                layer_name = self.layer_name_mapping[name]
                if layer_name in self.feature_layers:
                    features[layer_name] = x
        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate perceptual loss.

        Args:
            pred: predicted image tensor (B, C, H, W) in [0, 1]
            target: target image tensor (B, C, H, W) in [0, 1]

        Returns:
            Perceptual loss value
        """
        # Normalize to ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)

        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        # Extract features
        pred_features = self.extract_features(pred_norm)
        target_features = self.extract_features(target_norm)

        # Calculate loss
        loss = 0.0
        for layer_name in self.feature_layers:
            if layer_name in pred_features:
                loss += F.mse_loss(pred_features[layer_name], target_features[layer_name])

        return loss / len(self.feature_layers)

class GradientLoss(nn.Module):
    """Gradient loss for edge preservation."""

    def __init__(self):
        super().__init__()

        # Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def gradient(self, x: torch.Tensor) -> tuple:
        """Calculate image gradients."""
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return grad_x, grad_y

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate gradient loss.

        Args:
            pred: predicted image tensor (B, C, H, W) in [0, 1]
            target: target image tensor (B, C, H, W) in [0, 1]

        Returns:
            Gradient loss value
        """
        pred_grad_x, pred_grad_y = self.gradient(pred)
        target_grad_x, target_grad_y = self.gradient(target)

        loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
        return loss

# Factory function for creating losses
def create_loss(loss_type: str, **kwargs) -> nn.Module:
    """
    Create loss function by type.

    Args:
        loss_type: type of loss ('l1', 'mse', 'charbonnier', 'perceptual', 'gradient')
        **kwargs: additional arguments for loss function

    Returns:
        Loss function
    """
    loss_type = loss_type.lower()

    if loss_type == 'l1':
        return L1Loss(**kwargs)
    elif loss_type == 'mse':
        return MSELoss(**kwargs)
    elif loss_type == 'charbonnier':
        return CharbonnierLoss(**kwargs)
    elif loss_type == 'perceptual':
        return PerceptualLoss(**kwargs)
    elif loss_type == 'gradient':
        return GradientLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

# Default loss configuration for C2D-ISR
def get_default_loss_config() -> list:
    """Get default loss configuration for C2D-ISR training."""
    return [
        {'type': 'l1', 'weight': 1.0}
    ]

# Advanced loss configuration for better visual quality
def get_advanced_loss_config() -> list:
    """Get advanced loss configuration with multiple loss terms."""
    return [
        {'type': 'l1', 'weight': 0.8},
        {'type': 'charbonnier', 'weight': 0.1},
        {'type': 'gradient', 'weight': 0.1}
    ]
