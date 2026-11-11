# TransferGuard: Reproducibility Protocol & Environment Specification

> **Authors:** Arvind Yadav, Ghanishth Saini, Aakash Jagtap  
> **Repository:** https://github.com/arvindyadav186/TransferGuard-Detecting-and-Avoiding-Negative-Transfer-in-Lightweight-Vision-Models.git  

This document outlines the software environment, hardware testbeds, random seed controls, and exact command sequences required to reproduce all benchmark metrics in TransferGuard.

---

## 1. System Requirements

### 1.1 Hardware Specifications
- **GPU (Recommended for Full Training):** NVIDIA Tesla T4 (16 GB VRAM) or RTX 3080/4090/A100 with Tensor Core support.
- **CPU:** 4+ cores (Intel Core i5/i7/Xeon or AMD Ryzen).
- **RAM:** 16 GB system memory minimum.
- **Storage:** ~10 GB free disk space (datasets + virtual environment).

### 1.2 Software Dependencies
- **OS:** Linux (Ubuntu 20.04/22.04 LTS tested) or macOS / Windows via WSL2.
- **Python:** Version 3.9 through 3.12 (Python 3.10+ recommended).
- **PyTorch:** Version 2.0+ (CUDA 11.8 or CUDA 12.x enabled for GPU execution).
- **Torchvision:** Version 0.15+.

---

## 2. Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/arvindyadav186/TransferGuard-Detecting-and-Avoiding-Negative-Transfer-in-Lightweight-Vision-Models.git
cd TransferGuard-Detecting-and-Avoiding-Negative-Transfer-in-Lightweight-Vision-Models

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## 3. Automated Test Suite (Sanity Verification)

Execute the unit test suite on CPU to verify model hooks, gating controllers, CKA invariant properties, and training checkpoints:

```bash
pytest tests/ -v
```
**Expected outcome:** `12 passed in ~12s`.

---

## 4. Benchmark Reproduction Commands

All training runs use fixed random seeds (`seed: 42`) specified in `configs/*.yaml`.

### 4.1 Benchmark 1: SVHN on MobileNetV3-Small
```bash
# 1. Compute layer-wise CKA alignment curve
python scripts/run_cka_analysis.py --dataset svhn --model mobilenet_v3_small --device auto --subset 500

# 2. Scratch Baseline (Random Initialization)
python scripts/train_scratch.py --config configs/svhn_mobilenetv3.yaml --epochs 30 --device auto

# 3. Standard Full Fine-Tuning
python scripts/train_finetune.py --config configs/svhn_mobilenetv3.yaml --mode full --epochs 30 --device auto

# 4. Linear Probing (Frozen Backbone - Negative Transfer)
python scripts/train_finetune.py --config configs/svhn_mobilenetv3.yaml --mode linear_probe --epochs 30 --device auto

# 5. Proposed TransferGuard (Adaptive Gating & Routing)
python scripts/train_transferguard.py --config configs/svhn_mobilenetv3.yaml --threshold 0.70 --epochs 30 --device auto
```

### 4.2 Benchmark 2: SVHN on ShuffleNetV2 1.0x (Architecture Generalization)
```bash
python scripts/train_scratch.py --config configs/svhn_shufflenetv2.yaml --epochs 30 --device auto
python scripts/train_finetune.py --config configs/svhn_shufflenetv2.yaml --mode full --epochs 30 --device auto
python scripts/train_finetune.py --config configs/svhn_shufflenetv2.yaml --mode linear_probe --epochs 30 --device auto
python scripts/train_transferguard.py --config configs/svhn_shufflenetv2.yaml --threshold 0.70 --epochs 30 --device auto
```

### 4.3 Benchmark 3: CIFAR-100 on MobileNetV3-Small (Moderate Transfer)
```bash
python scripts/train_scratch.py --config configs/cifar100_mobilenetv3.yaml --epochs 30 --device auto
python scripts/train_finetune.py --config configs/cifar100_mobilenetv3.yaml --mode full --epochs 30 --device auto
python scripts/train_finetune.py --config configs/cifar100_mobilenetv3.yaml --mode linear_probe --epochs 30 --device auto
python scripts/train_transferguard.py --config configs/cifar100_mobilenetv3.yaml --threshold 0.70 --epochs 30 --device auto
```

### 4.4 Benchmark 4: Oxford Flowers-102 on MobileNetV3-Small (Data Scarcity)
```bash
python scripts/train_scratch.py --config configs/flowers102_mobilenetv3.yaml --epochs 30 --device auto
python scripts/train_finetune.py --config configs/flowers102_mobilenetv3.yaml --mode full --epochs 30 --device auto
python scripts/train_finetune.py --config configs/flowers102_mobilenetv3.yaml --mode linear_probe --epochs 30 --device auto
python scripts/train_transferguard.py --config configs/flowers102_mobilenetv3.yaml --threshold 0.70 --epochs 30 --device auto
```

---

## 5. Aggregation & Figure Generation

Once training runs are complete (or to inspect existing live logs):

```bash
# 1. Consolidate metrics and compute Transfer Gain (TG)
python scripts/evaluate_all.py

# 2. Render 300 DPI figures
python scripts/plot_results.py --output-dir results/figures
```

Outputs:
- Consolidated table: `results/benchmark_summary.csv` and `results/benchmark_summary.json`
- 300 DPI figures: `results/figures/*.png`

