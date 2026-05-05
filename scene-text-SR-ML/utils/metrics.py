import torch
import torch.nn.functional as F

def psnr(sr: torch.Tensor, hr: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    sr, hr: (B,C,H,W) in [0,1]
    """
    mse = F.mse_loss(sr, hr, reduction="none").mean(dim=(1,2,3))
    return 10.0 * torch.log10((data_range ** 2) / (mse + eps))

def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device="cpu"):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return kernel_2d

def ssim(sr: torch.Tensor, hr: torch.Tensor, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5):
    """
    Lightweight SSIM (per-image), grayscale-agnostic (averages across channels).
    sr, hr: (B,C,H,W) in [0,1]
    """
    device = sr.device
    k = _gaussian_kernel(window_size, sigma, device=device)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # depthwise conv per channel
    B, C, H, W = sr.shape
    k = k.repeat(C, 1, 1, 1)

    mu1 = F.conv2d(sr, k, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(hr, k, padding=window_size // 2, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(sr * sr, k, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(hr * hr, k, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(sr * hr, k, padding=window_size // 2, groups=C) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # average over channels and spatial dims
    return ssim_map.mean(dim=(1,2,3))
