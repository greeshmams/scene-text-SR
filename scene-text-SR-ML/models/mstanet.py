import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossChannelAttention(nn.Module):
    """
    Cross-channel attention conceptually aligned with your OIA attention selection:
    uses avg/max summaries and learns a spatial attention map. :contentReference[oaicite:17]{index=17}
    """
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        # x: (B,C,H,W)
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        a = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * a, a  # return attention map

class OIA(nn.Module):
    """
    Orientational-Invariant Attention module: (3x3, 1x3, 3x1) convs + attention. :contentReference[oaicite:18]{index=18}
    """
    def __init__(self, ch: int):
        super().__init__()
        self.conv_diag = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv_h = nn.Conv2d(ch, ch, (1,3), padding=(0,1))
        self.conv_v = nn.Conv2d(ch, ch, (3,1), padding=(1,0))
        self.reduce = nn.Conv2d(ch * 3, ch, 1)
        self.cca = CrossChannelAttention(ch)

    def forward(self, x):
        d = F.relu(self.conv_diag(x), inplace=True)
        h = F.relu(self.conv_h(x), inplace=True)
        v = F.relu(self.conv_v(x), inplace=True)
        f = F.relu(self.reduce(torch.cat([d,h,v], dim=1)), inplace=True)
        f, attn = self.cca(f)
        return f, attn

class SEChannelGate(nn.Module):
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(ch // r, 4), 1)
        self.fc2 = nn.Conv2d(max(ch // r, 4), ch, 1)

    def forward(self, x):
        g = x.mean(dim=(2,3), keepdim=True)
        g = F.relu(self.fc1(g), inplace=True)
        g = torch.sigmoid(self.fc2(g))
        return x * g, g

class MultiAtrousAttention(nn.Module):
    """
    Multi-atrous conv branches + channel gating + dynamic spatial attention. :contentReference[oaicite:19]{index=19}
    """
    def __init__(self, ch: int, rates=(1,6,12,18), reduction=16):
        super().__init__()
        self.rates = rates
        self.branches = nn.ModuleList([nn.Conv2d(ch, ch, 3, padding=r, dilation=r) for r in rates])
        self.fuse = nn.Conv2d(ch * len(rates), ch, 1)
        self.se = SEChannelGate(ch, r=reduction)
        self.spatial = nn.Conv2d(ch, 1, 7, padding=3)

    def forward(self, x):
        feats = [F.relu(b(x), inplace=True) for b in self.branches]   # list of (B,C,H,W)
        cat = self.fuse(torch.cat(feats, dim=1))                     # (B,C,H,W)
        cat, gate = self.se(cat)                                     # channel attention
        # dynamic spatial attention (single map)
        s = torch.sigmoid(self.spatial(cat))                         # (B,1,H,W)
        return cat * s, s, gate

class SeqResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        r = self.act(self.conv1(x))
        r = self.conv2(r)
        return x + r

class BiLSTM2D(nn.Module):
    """
    Optional BLSTM over width dimension, similar motivation to your sequential modeling. :contentReference[oaicite:20]{index=20}
    """
    def __init__(self, ch: int, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(input_size=ch, hidden_size=hidden, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden * 2, ch)

    def forward(self, x):
        # x: (B,C,H,W) -> sequence over W by pooling H
        B, C, H, W = x.shape
        seq = x.mean(dim=2).permute(0,2,1)  # (B,W,C)
        y, _ = self.lstm(seq)               # (B,W,2Hid)
        y = self.proj(y).permute(0,2,1).unsqueeze(2)  # (B,C,W)->(B,C,1,W)
        return x + y.expand(-1, -1, H, -1)

class MSTANet(nn.Module):
    """
    Malayalam Scene-text Attention Network: shallow + OIA + MultiAtrous + SeqRB + PixelShuffle.
    :contentReference[oaicite:21]{index=21}
    """
    def __init__(self, in_ch=3, feat=64, scale=3, num_oia=2, use_blstm=True):
        super().__init__()
        self.scale = scale
        self.shallow = nn.Conv2d(in_ch, feat, 3, padding=1)

        self.oias = nn.ModuleList([OIA(feat) for _ in range(num_oia)])
        self.multi = MultiAtrousAttention(feat, rates=(1,6,12,18), reduction=16)

        self.mix = nn.Conv2d(feat * 2, feat, 1)

        self.seq = nn.Sequential(SeqResidualBlock(feat), SeqResidualBlock(feat), SeqResidualBlock(feat))
        self.blstm = BiLSTM2D(feat) if use_blstm else nn.Identity()

        self.to_hr = nn.Conv2d(feat, in_ch * (scale ** 2), 3, padding=1)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, lr):
        f = F.relu(self.shallow(lr), inplace=True)

        attn_o_maps = []
        for oia in self.oias:
            f, a = oia(f)
            attn_o_maps.append(a)
        attn_o = torch.stack(attn_o_maps, dim=0).mean(dim=0)  # (B,1,H,W)

        f_s, attn_s, gate_s = self.multi(f)

        f_mix = F.relu(self.mix(torch.cat([f, f_s], dim=1)), inplace=True)
        f_mix = self.seq(f_mix)
        f_mix = self.blstm(f_mix)

        sr = self.ps(self.to_hr(f_mix)).clamp(0.0, 1.0)

        # return SR + attention maps for attention-guided losses
        return sr, {"attn_o": attn_o, "attn_s": attn_s, "gate_s": gate_s}
