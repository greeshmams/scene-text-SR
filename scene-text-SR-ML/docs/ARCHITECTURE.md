# Architecture Details

## GS²TNet: Generic Scene Text Triplet Network

### Overview
GS²TNet decomposes scene-text feature extraction into three complementary subspaces:

1. **Semantic CNN Branch**: 5×5 → 3×3 → 1×1 cascade for hierarchical semantic features
2. **Gabor-Modulated Branch (GCN)**: Learnable filters modulated by Gabor orientation filters
3. **Atrous Convolution Branch (SRC)**: Multi-rate dilated convolutions for disparity-aware context

### Gabor Convolution Configuration

| Parameter | Value |
|-----------|-------|
| Orientations (U) | 4: {20°, 40°, 80°, 160°} |
| Scales (V) | 4: {1, 2, 3, 4} |
| Gabor σ | {2, 4, 6, 8} |
| Channel multipliers (γ) | {8, 16} |

### Atrous Convolution Configuration

| Layer | 1 | 2 | 3 | 4 | 5 | 6 |
|-------|---|---|---|---|---|---|
| Kernel | 3×3 | 3×3 | 3×3 | 3×3 | 3×3 | 3×3 |
| Padding | 1 | 2 | 2 | 2 | 2 | 2 |
| Channels | 64 | 64 | 64 | 64 | 64 | 64 |
| Dilation | 1 | 3 | 4 | 6 | 12 | 18 |
| Receptive Field | 3×3 | 7×7 | 9×9 | 13×13 | 25×25 | 37×37 |

### Progressive Feature Fusion (PFF)
- Layer-by-layer fusion via 1×1 + 3×3 convolutions
- Outperforms GRL and GFF by 0.00–0.04 dB

---

## MSTANet: Malayalam Scene Text Attention Network

### Overview
Dual-attention architecture for Malayalam script (57 base characters, 540+ conjuncts):

1. **OIA Module**: Orientational-Invariant Attention with heterogeneous kernels
2. **Multi-Atrous Attention**: Scale-invariant context with channel + spatial gating
3. **Sequential Refinement**: SeqRB ×3 + Bidirectional LSTM

### OIA Configuration

| Component | Details |
|-----------|---------|
| Kernels | 3×3 (diagonal), 3×1 (horizontal), 1×3 (vertical) |
| Attention | Cross-channel across (c×w), (c×h), (h×w) |
| Optimal #OIA | 2 modules (+0.81 dB over baseline) |

### Multi-Atrous Attention Configuration

| Component | Details |
|-----------|---------|
| Dilation rates | {1, 6, 12, 18} |
| Channel reduction | ratio = 16 |
| Dynamic spatial | Softmax-gated attention |

### Loss Function

```
L_total = L_orientation + L_stereo
L_orient = λ₁·MSE(I_HR, I_SR) + λ₂·||Φ_orient(attn) - Φ_orient(I_HR)||₁
L_stereo = λ₁·MSE(I_HR, I_SR) + λ₂·||Φ_stereo(attn) - Φ_stereo(I_HR)||₁
```

where λ₁ = 0.5, λ₂ = 1.0

---

## Selective Patch-Pair Sampling

PSNR-guided augmentation (Algorithm 2 in manuscript):

1. Generate 10 candidate LR–HR patch pairs per image
2. Compute PSNR between bicubic-upsampled LR and HR patches
3. Retain only patches with PSNR < threshold (τ = 25.09 dB)
4. This filters out uninformative background patches

Results: +0.39 dB on SVT, +0.38 dB on ICDAR 2013
