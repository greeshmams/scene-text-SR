import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_gabor_kernel(ks: int, sigma: float, theta: float, lambd: float, gamma: float, psi: float, device):
    """
    Real Gabor kernel (2D). Used to modulate a learnable conv filter. :contentReference[oaicite:15]{index=15}
    """
    half = ks // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, device=device).float(),
        torch.arange(-half, half + 1, device=device).float(),
        indexing="ij"
    )
    # rotation
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)

    gb = torch.exp(-(x_theta**2 + (gamma**2) * y_theta**2) / (2 * sigma**2)) * torch.cos(2 * math.pi * x_theta / lambd + psi)
    gb = gb / (gb.abs().sum() + 1e-8)
    return gb

class GaborModulatedConv2d(nn.Module):
    """
    Learnable conv weights W are elementwise-multiplied by fixed Gabor kernels
    across multiple orientations/scales, then convolved.
    """
    def __init__(self, in_ch: int, out_ch: int, ks: int = 3, orientations=(20,40,80,160), scales=(1,2,3,4)):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ks = ks
        self.orientations = orientations
        self.scales = scales

        # base learnable filter
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, ks, ks) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        outs = []
        for deg in self.orientations:
            theta = math.radians(deg)
            for s in self.scales:
                g = make_gabor_kernel(
                    ks=self.ks, sigma=2.0, theta=theta, lambd=3.0 * s, gamma=0.5, psi=0.0, device=device
                )  # (ks,ks)
                g = g.view(1, 1, self.ks, self.ks)  # broadcast
                w = self.weight * g
                y = F.conv2d(x, w, self.bias, padding=self.ks // 2)
                outs.append(y)
        return torch.cat(outs, dim=1)  # concat along channel

class AtrousPyramid(nn.Module):
    def __init__(self, ch: int = 64, rates=(1,3,4,6,12,18)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(ch, ch, 3, padding=r, dilation=r) for r in rates
        ])
        self.fuse = nn.Conv2d(ch * len(rates), ch, 1)

    def forward(self, x):
        feats = [F.relu(conv(x), inplace=True) for conv in self.convs]
        return self.fuse(torch.cat(feats, dim=1))

class GS2TNet(nn.Module):
    """
    Generic Scene Text Triplet Network (GS2TNet) – core idea:
    CNN branch + Gabor branch + Atrous branch, progressively fused, then pixel shuffle.
    :contentReference[oaicite:16]{index=16}
    """
    def __init__(self, in_ch=3, feat=64, scale=3):
        super().__init__()
        self.scale = scale

        self.shallow = nn.Conv2d(in_ch, feat, 3, padding=1)

        # Branch 1: semantic CNN
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(feat, feat, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Branch 2: Gabor
        self.gabor = GaborModulatedConv2d(feat, feat, ks=3)
        self.gabor_post = nn.Sequential(
            nn.Conv2d(feat * (4 * 4), feat, 1),  # orientations * scales
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Branch 3: Atrous
        self.atrous = AtrousPyramid(ch=feat)

        # Progressive feature fusion (PFF)
        self.fuse1 = nn.Conv2d(feat * 3, feat, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(feat, feat, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Reconstruction
        self.to_hr = nn.Conv2d(feat, in_ch * (scale ** 2), 3, padding=1)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, lr):
        f0 = F.relu(self.shallow(lr), inplace=True)
        f1 = self.cnn_branch(f0)
        f2 = self.gabor_post(self.gabor(f0))
        f3 = self.atrous(f0)

        fused = F.relu(self.fuse1(torch.cat([f1, f2, f3], dim=1)), inplace=True)
        fused = self.refine(fused)

        out = self.ps(self.to_hr(fused))
        return out.clamp(0.0, 1.0)
