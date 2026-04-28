import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> float:
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    def gaussian(window_size: int, sigma: float) -> torch.Tensor:
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()
    
    def create_window(window_size: int, channel: int) -> torch.Tensor:
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    window = create_window(window_size, img1.size(1))
    if img1.is_cuda:
        window = window.cuda(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img1.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()

def batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    psnr_values = []
    ssim_values = []

    for i in range(pred.size(0)):
        psnr_val = calculate_psnr(pred[i], target[i])
        ssim_val = calculate_ssim(pred[i], target[i])
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)

    return np.mean(psnr_values), np.mean(ssim_values)

def tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
    if img.dim() == 4:
        img = img.squeeze(0)
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)

def save_image(img: torch.Tensor, path: str) -> None:
    import cv2
    img_np = tensor_to_numpy(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR.RGB2BGR)
    cv2.imwrite(path, img_bgr)
