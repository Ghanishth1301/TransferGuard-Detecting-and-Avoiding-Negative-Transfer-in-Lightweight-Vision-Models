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
# Run all 12 automated unit tests on CPU (CKA invariance, hooks, gates, resumable training, Drive sync)
pytest tests/ -v
```

### 3. Google Colab GPU Execution & Zero-Progress-Loss Persistence
For full-scale training on Tesla T4 GPU:
1. Open [`TransferGuard_Colab.ipynb`](./TransferGuard_Colab.ipynb) in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Mount Google Drive (`/content/drive/MyDrive/TransferGuard_Backups/`) to enable bidirectional checkpoint auto-sync.
4. All training commands support `--resume auto --drive-backup auto`: if your Colab instance times out, simply re-running the cell restores previous checkpoints from Drive and resumes without any lost compute.

---

## 🔬 Command Line Usage

### 1. Compute Layer-wise CKA Representation Similarity
```bash
python scripts/run_cka_analysis.py --dataset svhn --model mobilenet_v3_small --device auto --subset 500
```
Exports `results/cka/cka_mobilenet_v3_small_svhn.png` and structured metrics to `results/cka/cka_mobilenet_v3_small_svhn.json`.

### 2. Train Scratch Baseline
```bash
python scripts/train_scratch.py --config configs/svhn_mobilenetv3.yaml --epochs 30 --device auto --resume auto --drive-backup auto
```

### 3. Train Standard Fine-Tuning & Linear Probing Baselines
```bash
# Standard Full Fine-Tuning
python scripts/train_finetune.py --config configs/svhn_mobilenetv3.yaml --mode full --epochs 30 --device auto --resume auto --drive-backup auto

# Linear Probing (Frozen Backbone - Demonstrates Severe Negative Transfer)
python scripts/train_finetune.py --config configs/svhn_mobilenetv3.yaml --mode linear_probe --epochs 30 --device auto --resume auto --drive-backup auto
```

### 4. Train Proposed TransferGuard Adaptive Routing
```bash
python scripts/train_transferguard.py --config configs/svhn_mobilenetv3.yaml --threshold 0.70 --epochs 30 --device auto --resume auto --drive-backup auto
```

### 5. Evaluate Benchmark & Compute Transfer Gain
```bash
python scripts/evaluate_all.py
```
Automatically detects live GPU history files in `results/`, computes Transfer Gain ($TG = Acc_{transfer} - Acc_{scratch}$), and outputs `results/benchmark_summary.csv` and `results/benchmark_summary.json`.

### 6. Generate 300 DPI Publication & Presentation Figures
```bash
python scripts/plot_results.py --output-dir results/figures
```
Dynamically detects real GPU logs and checkpoints to render:
1. `transfer_gain_comparison.png` (Positive vs Negative Transfer zones)
2. `cka_divergence_curve.png` (Layer-wise CKA alignment cliff)
3. `transferguard_svhn_routing_profile.png` (Dynamic stage decisions: Freeze, Bypass, Tune)
4. `training_convergence_comparison.png` (Live epoch-by-epoch validation trajectories)

### 7. Interactive Live Presentation Demo
```bash
python scripts/demo_live.py
```
Simulates real-time inference with colored ASCII meters showing stage gating scores and dynamic routing actions for oral defense presentations.

---

## 📊 Live GPU Empirical Benchmark Results (SVHN / MobileNetV3)

| Contender | Backbone Initialization | Trainable Weights | SVHN Top-1 Acc | Transfer Gain ($TG$) | Negative Transfer? | Convergence Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train from Scratch** | Random (He Normal) | 100% (~2.54M) | **96.43%** | 0.00% (Baseline) | N/A | Slow (96% at Epoch 18) |
| **Linear Probing** | ImageNet-1k | **~0.6% (~15K)** | **65.53%** | **-30.90%** | ⚠️ **Severe Collapse** | Underfits completely |
| **Standard Fine-Tuning** | ImageNet-1k | 100% (~2.54M) | **97.12%** | +0.69% | ✅ Positive Transfer | Moderate (96% at Epoch 6) |
| **TransferGuard (Ours)** | ImageNet-1k | **Dynamic (~40%)** | **96.62%** | +0.19% | ✅ Positive Transfer | 🚀 **Fastest (96% at Epoch 3)** |

### Key Experimental Insights:
1. **Negative Transfer is Confirmed:** Naive transfer (Linear Probing) catastrophically collapses by **-30.90%**, proving ImageNet natural features cannot directly solve street numbers without adaptation.
2. **Autonomous Gate Specialization:** TransferGuard's micro-gates autonomously assigned `1.00 (Freeze)` to universal early/mid features and `0.00 (Bypass)` to conflicting deep features, matching computer vision transfer theory.
3. **Superior Sample Efficiency:** TransferGuard reached **96.0% accuracy in just 3 epochs** (6× faster than Scratch, 2× faster than Full Fine-Tuning) while keeping ~60% of the backbone frozen.

---

## 📜 Documentation Links
- [Detailed Project Specification](docs/PROJECT_SPEC.md)
- [Architecture & Design Details](docs/ARCHITECTURE_AND_DESIGN.md)
- [Experiment & Benchmarking Plan](docs/EXPERIMENT_PLAN.md)
- [Beginner-Friendly System Walkthrough & Dataflow Guide](docs/SYSTEM_WALKTHROUGH_AND_DATAFLOW.md)
- [Presentation Defense Guide & Speaker Scripts](docs/PRESENTATION_DEFENSE_GUIDE.md)

