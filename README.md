# TransferGuard: Detecting and Avoiding Negative Transfer in Lightweight Vision Models

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **Computer Vision Course Project**  
> **Team Members:**
> - **Ghanishth Saini** (Roll: 24121)
> - **Arvind Yadav** (Roll: 24059)
> - **Aakash Jagtap** (Roll: 24157)

---

## 📌 Project Overview

Transfer learning from ImageNet is a cornerstone of modern deep computer vision. However, when adapting lightweight vision architectures (e.g., **MobileNetV3**, **ShuffleNetV2**) to target domains with severe distribution divergence (e.g., **SVHN** house numbers), transfer learning often induces **Negative Transfer**—where fine-tuning pre-trained weights produces lower accuracy than training from scratch ($TG < 0$).

Because lightweight models have restricted parameter capacity and narrow bottleneck channels, deep representations specialized for natural objects interfere with structured target representations, distorting early generic filters during backpropagation.

**TransferGuard** resolves this problem through:
1. **Layer-wise Representational Alignment:** Leveraging Centered Kernel Alignment (CKA) to identify the exact depth where source and target representations diverge.
2. **TransferGuard Gating & Dynamic Routing:** An adaptive module that automatically freezes or bypasses harmful deep layers, preserving generic early features and turning negative transfer into positive transfer gain.

---

## 🗂️ Repository Structure

```
.
├── 12_Ghanishth_Saini_24121.pptx     # Submitted Project Presentation
├── TransferGuard_Colab.ipynb         # One-click full GPU training notebook for Google Colab
├── README.md                         # Project documentation and quickstart
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
├── configs/                          # Experiment configuration YAMLs
│   ├── default.yaml
│   ├── svhn_mobilenetv3.yaml
│   ├── svhn_shufflenetv2.yaml
│   ├── cifar100_mobilenetv3.yaml
│   └── flowers102_mobilenetv3.yaml
├── docs/                             # Full architectural & academic documentation
│   ├── PROJECT_SPEC.md               # Theoretical foundation & mathematical formulations
│   ├── ARCHITECTURE_AND_DESIGN.md    # Detailed module design and dataflows
│   ├── EXPERIMENT_PLAN.md            # Systematic benchmarking plan & hypotheses
│   └── PRESENTATION_DEFENSE_GUIDE.md # Slide-by-slide speaker script & examiner Q&A
├── src/                              # Core implementation package
│   ├── models/                       # Backbones, wrappers, and TransferGuard module
│   │   ├── mobilenet_v3.py
│   │   ├── shufflenet_v2.py
│   │   ├── transferguard_gate.py
│   │   └── adaptive_routing.py
│   ├── metrics/                      # CKA and transfer gain evaluation
│   │   ├── cka.py
│   │   ├── transfer_gain.py
│   │   └── transferability.py
│   ├── data/                         # Dataset loaders (SVHN, CIFAR-100, Flowers-102)
│   │   ├── dataset_loader.py
│   │   └── transforms.py
│   ├── training/                     # Unified training engine
│   │   ├── trainer.py
│   │   ├── loss.py
│   │   └── callbacks.py
│   └── utils/                        # Visualization and config helpers
│       ├── visualization.py
│       └── config.py
├── scripts/                          # Entrypoint CLI scripts
│   ├── run_cka_analysis.py           # Layer-wise CKA matrix computation
│   ├── train_scratch.py              # Baseline: Train from scratch
│   ├── train_finetune.py             # Baseline: Fine-tune & linear probe
│   ├── train_transferguard.py        # Proposed: TransferGuard adaptive routing
│   ├── evaluate_all.py               # Comparative Transfer Gain generator
│   └── plot_results.py               # Generates 300 DPI figures for presentation
└── tests/                            # Unit tests and CPU sanity checks
    ├── test_cka.py
    ├── test_models.py
    └── test_pipeline_dryrun.py
```

---

## 🚀 Quickstart Guide

### 1. Clone & Environment Setup
```bash
# Clone the repository
git clone https://github.com/arvindyadav186/TransferGuard-Detecting-and-Avoiding-Negative-Transfer-in-Lightweight-Vision-Models.git
cd TransferGuard-Detecting-and-Avoiding-Negative-Transfer-in-Lightweight-Vision-Models

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### 2. Fast CPU Sanity Check (Runs in < 60 seconds)
```bash
# Run automated tests on CPU with synthetic mini-batches
pytest tests/
```

### 3. Google Colab GPU Execution
For full-scale training on 73K SVHN images and 50K CIFAR-100 images:
1. Open [`TransferGuard_Colab.ipynb`](./TransferGuard_Colab.ipynb) in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run all cells to reproduce the complete experimental benchmark and export high-resolution presentation figures.

---

## 🔬 Command Line Usage

### 1. Compute Layer-wise CKA Representation Similarity
```bash
python scripts/run_cka_analysis.py --dataset svhn --model mobilenet_v3_small --device cpu --subset 500
```

### 2. Train Scratch Baseline
```bash
python scripts/train_scratch.py --config configs/svhn_mobilenetv3.yaml --epochs 50 --device auto
```

### 3. Train Standard Fine-Tuning Baseline
```bash
python scripts/train_finetune.py --config configs/svhn_mobilenetv3.yaml --mode full --epochs 30 --device auto
```

### 4. Train Proposed TransferGuard Adaptive Routing
```bash
python scripts/train_transferguard.py --config configs/svhn_mobilenetv3.yaml --threshold 0.65 --epochs 30 --device auto
```

### 5. Generate Presentation Visualizations
```bash
python scripts/plot_results.py --output_dir results/figures
```

---

## 📊 Summary of Theoretical Findings

| Regime | Source Domain | Target Domain | Standard Transfer Gain ($TG$) | TransferGuard Gain ($TG$) | CKA Behavior in Deep Layers |
|---|---|---|:---:|:---:|---|
| **Negative Transfer** | ImageNet | **SVHN** (Digits) | **Negative ($TG < 0$)** | **Positive ($TG > 0$)** | Sharp drop ($CKA < 0.35$) |
| **Mild Transfer** | ImageNet | **CIFAR-100** | Positive ($TG > 0$) | Enhanced ($TG \uparrow$) | Moderate ($CKA \approx 0.60$) |
| **Data-Scarce Transfer** | ImageNet | **Flowers-102** | Strong Positive ($TG \gg 0$) | Preserved ($TG \gg 0$) | High ($CKA > 0.80$) |

---

## 📜 Documentation Links
- [Detailed Project Specification](docs/PROJECT_SPEC.md)
- [Architecture & Design Details](docs/ARCHITECTURE_AND_DESIGN.md)
- [Experiment & Benchmarking Plan](docs/EXPERIMENT_PLAN.md)
- [Presentation Defense Guide & Speaker Scripts](docs/PRESENTATION_DEFENSE_GUIDE.md)

