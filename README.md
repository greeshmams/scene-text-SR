# Orientation and Disparity Aware Scene Text Super-Resolution with Triplet Convolution and Specialized Malayalam Text Attention

##  Overview

Scene text image super-resolution (STISR) is critical for enhancing low-resolution text in real-world images for downstream recognition. This work presents two complementary architectures:

| Model | Purpose | Key Innovation | Parameters |
|-------|---------|----------------|------------|
| **GS²TNet** | Generic scene-text SR | Triplet convolution (CNN + Gabor + Atrous) with Progressive Feature Fusion | <600K |
| **MSTANet** | Malayalam scene-text SR | Orientational-Invariant Attention (OIA) + Multi-Atrous Attention + BiLSTM | ~920K |

Additionally, we propose **Selective Patch-Pair Sampling** — a PSNR-guided data augmentation strategy that retains only informative LR–HR patches for training.

### Key Results

| Dataset | Model | PSNR (dB) | SSIM | Scale |
|---------|-------|-----------|------|-------|
| ICDAR 2015 | GS²TNet | **26.65** | **0.895** | ×3 |
| TextZoom-Hard | GS²TNet | **20.45** | 0.753 | ×3 |
| ML-SceneText | MSTANet | **19.86** | **0.731** | ×3 |
| ML-SceneText | MSTANet | **22.56** | **0.780** | ×2 |

---

## Architecture

### GS²TNet — Generic Scene Text Triplet Network

```
Input LR → Shallow Conv → ┬─ CNN Branch (5×5→3×3→1×1)
                           ├─ Gabor Branch (GoF: θ={20°,40°,80°,160°}, σ={1,2,3,4})
                           └─ Atrous Branch (DR={1,3,4,6,12,18}, RF: 3×3→37×37)
                           → Progressive Feature Fusion → Pixel Shuffle → HR Output
```

### MSTANet — Malayalam Scene Text Attention Network

```
Input LR → Shallow Conv → OIA Module (3×3 + 1×3 + 3×1 + Cross-Channel Attention)
                         → Multi-Atrous Attention (rates={1,6,12,18} + Channel/Spatial Gating)
                         → Concat + Mix → SeqRB ×3 → BiLSTM → Pixel Shuffle → HR Output
```

---

## Repository Structure

```
scene_text-SR/
├── models/
│   ├── gs2tnet.py          # GS²TNet: Triplet convolution network
│   ├── mstanet.py          # MSTANet: Malayalam attention network
│   └── __init__.py
├── data/
│   ├── patchpair.py        # Selective patch-pair sampling (Algorithm 2)
│   └── __init__.py
├── utils/
│   ├── metrics.py          # PSNR and SSIM computation
│   └── __init__.py
├── train.py                # Training script for both models
├── requirements.txt        # Dependencies
├── README.md               # This file
├── LICENSE                 # MIT License
└── docs/
    └── ARCHITECTURE.md     # Detailed architecture description
```

---

##  Installation

### Prerequisites

- Python ≥ 3.9
- CUDA ≥ 11.8 (for GPU training)
- NVIDIA GPU with ≥ 16 GB VRAM (Tesla V100 32 GB recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/greeshmams/scene_text-SR.git
cd scene_text-SR

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥ 2.1 | Deep learning framework |
| torchvision | ≥ 0.16 | Image transforms |
| numpy | latest | Numerical computation |
| opencv-python | latest | Image I/O |
| Pillow | latest | Image loading |
| tqdm | latest | Progress bars |

---

## Datasets

### Supported Datasets

| Dataset | Images | Type | Download |
|---------|--------|------|----------|
| TextZoom | 17,367 | Real LR–HR pairs | [Link](https://github.com/JasonBoy1/TextZoom) |
| ICDAR 2015 | 2,077 | Street-view | [Link](https://rrc.cvc.uab.es/) |
| ICDAR 2013 | 233 | Scene images | [Link](https://rrc.cvc.uab.es/) |
| SVT | 647 | Street-view | [Link](http://vision.ucsd.edu/~kai/svt/) |
| **ML-SceneText** | **5,000** | **Malayalam outdoor** | https://drive.google.com/drive/folders/1EweqOSDRSs7uUW3ET84UwDFPX4ZTWZsg?usp=drive_link |

### ML-SceneText Dataset

Our newly constructed Malayalam scene-text dataset:
- **5,000** outdoor scene-text images (4,000 train / 1,000 test)
- Captured with **Nikon D3400** at **110–300 mm** focal length
- Diverse sizes, fonts, and backgrounds
- Malayalam script: 57 base characters, 540+ conjunct forms

---

##  Training

### GS²TNet (Generic Scene Text)

```bash
python train.py \
    --hr_dir /path/to/icdar2015/train/hr \
    --model gs2tnet \
    --scale 3 \
    --epochs 100 \
    --batch 16 \
    --lr 1e-4 \
    --patch_hr 96 \
    --save checkpoints/gs2tnet
```

### MSTANet (Malayalam Scene Text)

```bash
python train.py \
    --hr_dir /path/to/ml_scenetext/train \
    --model mstanet \
    --scale 3 \
    --epochs 100 \
    --batch 16 \
    --lr 1e-4 \
    --patch_hr 96 \
    --use_attention_loss \
    --save checkpoints/mstanet
```

### Training Configuration

| Parameter | GS²TNet | MSTANet |
|-----------|---------|---------|
| Optimizer | Adam (β₁=0.9, β₂=0.999) | Adam (β₁=0.9, β₂=0.999) |
| Learning rate | 1×10⁻⁴ | 1×10⁻⁴ |
| Batch size | 16 | 16 |
| Epochs | 100 | 100 |
| Loss | L1 | MSE + λ₁·L_orient + λ₂·L_stereo |
| Loss weights | — | λ₁=0.5, λ₂=1.0 |
| Scale factors | ×2, ×3 | ×2, ×3 |

---

## Key Algorithms

### 1. Gabor-Modulated Convolution (GCN)

Each learnable filter is element-wise multiplied by Gabor orientation kernels:

```
GF_{i,uv} = gf_{i,0} ⊙ G(u,v)
```

Configuration: U=4 orientations (20°, 40°, 80°, 160°), V=4 scales

### 2. Atrous Convolution Network (ACN)

Six 3×3 atrous layers with dilation rates {1, 3, 4, 6, 12, 18}, enlarging the receptive field from 3×3 to 37×37.

### 3. Orientational-Invariant Attention (OIA)

Three heterogeneous kernels (3×3, 3×1, 1×3) capture diagonal, horizontal, and vertical features, followed by cross-channel attention across (c×w), (c×h), and (h×w) dimensions.

### 4. Selective Patch-Pair Sampling

PSNR-guided augmentation that generates 10 candidate LR–HR patch pairs per image and retains only those below a PSNR threshold (τ = 25.09 dB), enriching training with text-critical samples.

---

## Evaluation

PSNR and SSIM are computed on the Y channel in YCbCr space. Recognition accuracy is measured using pre-trained ASTER, MORAN, and CRNN recognisers (without fine-tuning).

---



---

## Contact

- **Greeshma M. S.** : greeshmams.r@gmail.com
- School of Computer Sciences, Mahatma Gandhi University, Kottayam, Kerala 686560, India

---


## Acknowledgements

This work was conducted at Mahatma Gandhi University and Cochin University of Science and Technology, Kerala, India.
