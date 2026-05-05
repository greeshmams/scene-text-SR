from dataclasses import dataclass
import random
import numpy as np
import torch
import torch.nn.functional as F

def _psnr_np(a: np.ndarray, b: np.ndarray, data_range=1.0, eps=1e-8) -> float:
    mse = np.mean((a - b) ** 2)
    return 10.0 * np.log10((data_range ** 2) / (mse + eps))

@dataclass
class SelectivePatchPairConfig:
    scale: int = 3
    patch_hr: int = 96              # HR patch size
    aug_level: int = 8              # number of candidate patches per image
    psnr_threshold: float = 28.0    # keep "hard" pairs below this threshold

class SelectivePatchPairSampler:
    """
    Implements Algorithm 2 conceptually: sample multiple LR/HR patches and keep only
    informative pairs based on a PSNR criterion. :contentReference[oaicite:14]{index=14}
    """
    def __init__(self, cfg: SelectivePatchPairConfig):
        self.cfg = cfg

    @torch.no_grad()
    def sample(self, hr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hr: (C,H,W) in [0,1]
        returns (lr_patch, hr_patch)
        """
        c, H, W = hr.shape
        s = self.cfg.scale
        ph = self.cfg.patch_hr
        pl = ph // s

        # create LR by bicubic downsample
        hr_b = hr.unsqueeze(0)
        lr_b = F.interpolate(hr_b, scale_factor=1.0/s, mode="bicubic", align_corners=False)
        lr = lr_b.squeeze(0)

        best = None

        for _ in range(self.cfg.aug_level):
            y_hr = random.randint(0, H - ph)
            x_hr = random.randint(0, W - ph)
            hr_patch = hr[:, y_hr:y_hr+ph, x_hr:x_hr+ph]

            y_lr, x_lr = y_hr // s, x_hr // s
            lr_patch = lr[:, y_lr:y_lr+pl, x_lr:x_lr+pl]

            # compute PSNR between bicubic-upsampled LR patch and HR patch
            up = F.interpolate(lr_patch.unsqueeze(0), scale_factor=s, mode="bicubic", align_corners=False).squeeze(0)

            a = up.permute(1,2,0).cpu().numpy()
            b = hr_patch.permute(1,2,0).cpu().numpy()
            p = _psnr_np(a, b)

            if p <= self.cfg.psnr_threshold:
                return lr_patch, hr_patch  # accept first "hard" pair

            # if none meet threshold, keep the hardest (lowest PSNR)
            if best is None or p < best[0]:
                best = (p, lr_patch, hr_patch)

        assert best is not None
        _, lr_patch, hr_patch = best
        return lr_patch, hr_patch
