# MS-AFR-Net: Multi-Scale Attention-Driven Fingerprint Recognition Network

> Reproducing AFR-Net and proposing MS-AFR-Net — a multi-scale extension achieving **53.5% EER reduction** over the baseline.

## Overview

This project reproduces the [AFR-Net](https://arxiv.org/abs/2211.13897) architecture (Grosz & Jain, 2024) for universal fingerprint recognition and proposes **MS-AFR-Net**, which extends it with:

- **Multi-Scale Feature Pyramid** — tokenizes Conv3 (784 tokens), Conv4 (196 tokens), and Conv5 (49 tokens)
- **Cross-Scale Attention** — 6 Transformer layers process all 1,029 tokens jointly
- **Adaptive Scale Fusion Gate** — dynamically weights macro, meso, and micro features per input

## Key Results

| Metric | AFR-Net | MS-AFR-Net | Improvement |
|--------|---------|------------|-------------|
| **EER (7 Primary)** | 41.29% | **19.22%** | ↓53.5% |
| **TAR@FAR=0.1%** | 0.003 | **0.030** | ↑9.8× |
| **Rank-1 Accuracy** | 7.71% | **40.22%** | ↑5.2× |
| **Parameters** | 47.3M | **37.7M** | ↓20.3% |

Both models trained on **403,222 images** (31% of original paper's 1.3M), 75 epochs, identical configuration.

## Architecture

### AFR-Net (Baseline)
```
Input → STN (5-layer, 4 params) → ResNet-50 → Conv4 → 12 ViT layers → z_a (384-d)
                                             → Conv5 → GAP → FC → z_c (384-d)
                                             → z = [z_c; z_a] ∈ ℝ^768
```

### MS-AFR-Net (Proposed)
```
Input → STN (3-layer + BN, 6 params) → ResNet-50 → Conv3 (784 tokens)
                                                   → Conv4 (196 tokens) → Feature Pyramid
                                                   → Conv5 (49 tokens)    → 1,029 tokens
                                                   → 6 Cross-Scale Attention layers
                                                   → Adaptive Fusion Gate → z_ms (384-d)
                                                   → Conv5 → GAP → FC+BN → z_c (384-d)
                                                   → z = [z_c; z_ms] ∈ ℝ^768
```

## Repository Structure

```
├── ms_afrnet_training_kaggle.py     # Full training script (both AFR-Net & MS-AFR-Net)
├── afrnet_evaluation.py             # All-in-one evaluation (embedding + metrics + plots)
├── ms_afrnet_eval_part1.py          # MS-AFR-Net embedding extraction (Kaggle Part 1)
├── ms_afrnet_eval_part2.py          # MS-AFR-Net metrics & plots (Kaggle Part 2)
├── dataset_analysis_kaggle.py       # Dataset statistics and analysis
├── generate_synthetic_fingerprints.py # Synthetic fingerprint generation
├── report.tex                       # IEEE-format LaTeX report
├── references.bib                   # BibTeX references
├── AFRNET Results/                  # AFR-Net evaluation outputs (plots, JSON)
├── MS-AFRNET Results/               # MS-AFR-Net evaluation outputs (plots, JSON)
├── base_architecture.png            # AFR-Net architecture diagram
├── msafrnet_architecture_corrected.png # MS-AFR-Net architecture diagram
├── afrnet_training_log.csv          # AFR-Net training logs
├── ms_afrnet_training_log.csv       # MS-AFR-Net training logs
└── README.md
```

## Setup & Dependencies

### Requirements
- Python 3.10+
- PyTorch 2.x with CUDA support
- torchvision
- NumPy, Pillow, matplotlib, scipy

```bash
pip install torch torchvision numpy pillow matplotlib scipy
```

### Hardware
- Training: NVIDIA Tesla T4 (16 GB) on Kaggle
- Inference: ~38ms per image (batch=64)

## Training

### Kaggle Datasets Required
| Input Name | Kaggle Slug |
|------------|-------------|
| Dataset1 (Real) | `kushp3690/afr-training-real` |
| Dataset2 (Synthetic) | `kushp3690/afr-training-synthetic` |
| Dataset3 (Misc) | `kushp3690/afr-training-misc` |
| Dataset4 (Validation) | `kushp3690/l3-sf-v2` |

### Run Training
Upload `ms_afrnet_training_kaggle.py` to a Kaggle notebook with GPU (T4) and run all cells.

**Training Configuration:**
- Optimizer: AdamW (LR=1e-4, WD=2e-5)
- Loss: ArcFace (s=64, m=0.5)
- Scheduler: Polynomial LR decay with warmup
- Epochs: 75, Batch size: 64, Mixed precision (FP16)
- Augmentations: RandomRotation, RandomAffine, ColorJitter, GaussianBlur, RandomErasing, etc.

## Evaluation

### Test Datasets Required
| Input Name | Kaggle Slug |
|------------|-------------|
| MOLF Testing | `kushp3690/molf-testing-dataset` |
| PolyU Testing | `kushp3690/polyu-testing-datasets` |
| ISPFDv2 Testing | `kushp3690/ispfdv2-testing-dataset` |
| FVC + SD302 | `kushp3690/afr-training-real` |
| MS-AFR-Net Checkpoint | `talakshrabari/ms-afrnet-best-version2` |
| AFR-Net Checkpoint | `talakshrabari/afrnet-best-fromscratch` |

### Option 1: All-in-One
```bash
# Upload afrnet_evaluation.py to Kaggle notebook
# Attach model checkpoint + test datasets
# Run — generates embeddings, metrics, and plots in one go
```

### Option 2: Two-Part (for large datasets)
```bash
# Part 1: Extract embeddings (saves .npy files)
# Part 2: Compute metrics + generate plots from saved embeddings
```

### Evaluation Metrics
- **EER** (Equal Error Rate) — authentication performance
- **TAR@FAR=0.1%** and **TAR@FAR=0.01%** — true accept rate at strict thresholds
- **Rank-1 / Rank-5** — closed-set identification accuracy
- **DET Curves** — FAR vs FRR trade-off
- **CMC Curves** — cumulative match characteristic

## Test Benchmarks (11 Datasets)

| Dataset | Type | Subjects | Images |
|---------|------|----------|--------|
| FVC 2002 DB1A | Live Optical | 100 | 800 |
| FVC 2002 DB2A | Live Optical | 100 | 800 |
| FVC 2002 DB3A | Live Capacitive | 100 | 800 |
| FVC 2004 DB1A | Live Optical | 100 | 800 |
| FVC 2004 DB2A | Live Optical | 100 | 800 |
| FVC 2004 DB3A | Live Thermal | 100 | 800 |
| PolyU Contact | Live Optical | 336 | 2,976 |
| PolyU Contactless | Contactless 2D | 6 | 2,976 |
| PolyU Processed | Processed C'less | 6 | 2,976 |
| ISPFDv2 | Fingerphoto | 75 | 19,200 |
| NIST SD302 | Mixed Rolled+Latent | 20 | 7,140 |

## Base Paper

> S. A. Grosz and A. K. Jain, "AFR-Net: Attention-Driven Fingerprint Recognition Network," *IEEE Transactions on Biometrics, Behavior, and Identity Science*, 2024. [arXiv:2211.13897](https://arxiv.org/abs/2211.13897)

## Authors

- Kush Patel
- Talak Shrabari

## License

This project is for academic purposes only. The datasets used are subject to their respective license agreements.
