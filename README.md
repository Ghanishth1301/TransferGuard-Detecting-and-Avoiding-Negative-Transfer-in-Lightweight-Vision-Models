# TransferGuard: Detecting and Avoiding Negative Transfer in Lightweight Vision Models

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**Authors:** Arvind Yadav, Ghanishth Saini, Aakash Jagtap  
**Repository:** [GitHub](https://github.com/arvindyadav186/TransferGuard-Detecting-and-Avoiding-Negative-Transfer-in-Lightweight-Vision-Models.git)  

---

## 📌 Overview

Transfer learning from ImageNet is a cornerstone of modern computer vision. However, when adapting lightweight vision architectures (e.g., **MobileNetV3**, **ShuffleNetV2**) to target domains with severe distribution divergence (e.g., **SVHN** house numbers), transfer learning often induces **Negative Transfer**—where standard fine-tuning or feature reuse performs worse than training from scratch with random initialization ($TG < 0$).

Because lightweight edge models have compact parameter budgets and narrow bottleneck channels, deep representations specialized for natural objects conflict with structured target representations, distorting early generic filters during backpropagation.

**TransferGuard** resolves this problem through:
1. **Layer-wise Representational Alignment:** Leveraging Centered Kernel Alignment (CKA) to identify the exact depth where source and target representations diverge.
2. **TransferGuard Gating & Dynamic Routing:** An adaptive module that automatically evaluates layer compatibility, locking pre-trained weights in transferable stages and bypassing or adapting conflicting deep stages.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                TRANSFERGUARD WORKFLOW                                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
  Input Image [B, 3, 224, 224]
        │
        ▼
  Lightweight Backbone (MobileNetV3 / ShuffleNetV2)
        │
        ├── Stem (Conv 3x3)      ──► Forward Hook ──► Micro-Gate ──► S_stem   (Freeze: τ ≥ 0.70)
        ├── Stage Early          ──► Forward Hook ──► Micro-Gate ──► S_early  (Freeze: τ ≥ 0.70)
        ├── Stage Mid-1          ──► Forward Hook ──► Micro-Gate ──► S_mid1   (Freeze: τ ≥ 0.70)
        ├── Stage Mid-2          ──► Forward Hook ──► Micro-Gate ──► S_mid2   (Tune:   0.35 < τ < 0.70)
        └── Stage Deep           ──► Forward Hook ──► Micro-Gate ──► S_deep   (Bypass: τ ≤ 0.35)
                                                             │
                                                             ▼
                                                Dynamic Routing Controller
                                          [Preserves generic early filters,
                                        eliminates deep gradient interference]
```

---

## 📊 Consolidated Live GPU Empirical Benchmark Results

All experiments were executed on Tesla T4 GPUs with mixed precision (AMP) over 30 epochs per contender. Results are automatically aggregated from live training logs:

| Dataset & Task | Contender | Backbone Init | Top-1 Accuracy | Transfer Gain ($TG$) | Negative Transfer? | Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **SVHN**<br>*(MobileNetV3-Small)*<br>*Severe Domain Mismatch* | Train from Scratch | Random | **96.43%** | 0.00% | Baseline | He normal initialization baseline |
| | Linear Probing | ImageNet-1k | **65.53%** | **-30.90%** | ⚠️ **Severe Collapse** | Deep features conflict with digit morphology |
| | Standard Fine-Tuning | ImageNet-1k | **97.12%** | +0.69% | ✅ Positive Transfer | Full backprop adapts representation |
| | **TransferGuard (Ours)** | ImageNet-1k | **97.14%** | **+0.71%** | ✅ **Positive Transfer** | **Peak accuracy; 96% reached in 3 epochs** |
| **SVHN**<br>*(ShuffleNetV2 1.0x)*<br>*Architecture Generalization* | Train from Scratch | Random | **96.01%** | 0.00% | Baseline | Random initialization baseline |
| | Linear Probing | ImageNet-1k | **46.01%** | **-50.00%** | ⚠️ **Severe Collapse** | Catastrophic channel shuffle mismatch |
| | Standard Fine-Tuning | ImageNet-1k | **96.96%** | +0.95% | ✅ Positive Transfer | Recovers through full weight adaptation |
| | **TransferGuard (Ours)** | ImageNet-1k | **97.03%** | **+1.02%** | ✅ **Positive Transfer** | **Peak accuracy; Stem frozen, deep bypassed** |
| **CIFAR-100**<br>*(MobileNetV3-Small)*<br>*Moderate Domain Shift* | Train from Scratch | Random | **63.17%** | 0.00% | Baseline | Random initialization baseline |
| | Linear Probing | ImageNet-1k | **70.17%** | +7.00% | ✅ Positive Transfer | Natural object features transfer well |
| | Standard Fine-Tuning | ImageNet-1k | **81.86%** | +18.69% | ✅ Positive Transfer | High benefit from pre-trained weights |
| | **TransferGuard (Ours)** | ImageNet-1k | **79.52%** | **+16.35%** | ✅ **Positive Transfer** | Strong positive transfer with selective routing |
| **Oxford Flowers-102**<br>*(MobileNetV3-Small)*<br>*High Data Scarcity* | Train from Scratch | Random | **1.43%** | 0.00% | Baseline | Extreme overfitting (10 samples/class) |
| | Linear Probing | ImageNet-1k | **74.81%** | +73.38% | ✅ Positive Transfer | ImageNet features essential for small data |
| | Standard Fine-Tuning | ImageNet-1k | **81.39%** | +79.96% | ✅ Positive Transfer | Maximum positive transfer gain |
| | **TransferGuard (Ours)** | ImageNet-1k | **80.22%** | **+78.79%** | ✅ **Positive Transfer** | Robust transfer preserving pre-trained filters |

### Key Experimental Insights:
1. **Negative Transfer Confirmed Across Multiple Architectures:** On SVHN, linear probing collapses by **-30.90%** on MobileNetV3 and **-50.00%** on ShuffleNetV2, proving that ImageNet natural feature representations directly harm structured target tasks when frozen without adaptive gating.
2. **TransferGuard Outperforms Scratch & Fine-Tuning on SVHN:** On both MobileNetV3 (**97.14%**) and ShuffleNetV2 (**97.03%**), TransferGuard achieves the highest top-1 accuracy while freezing early universal filters and bypassing conflicting deep layers.
3. **Cross-Architecture Generalization:** TransferGuard functions identically on depthwise inverted residual blocks (MobileNetV3) and channel-split shuffle units (ShuffleNetV2), proving architectural universality.
4. **Faster Convergence & Efficiency:** On SVHN, TransferGuard reaches **96.0% accuracy in just 3 epochs** (6× faster than Scratch, 2× faster than Full Fine-Tuning) with ~60% of backbone layers frozen during training.

---

## 🎨 Publication Figures (300 DPI)

All figures are automatically generated from live experiment data:

- **Transfer Gain Comparison:** [`results/figures/transfer_gain_comparison.png`](results/figures/transfer_gain_comparison.png) — 12-bar comparison highlighting negative transfer (red) and positive transfer (green) zones.
- **CKA Representational Divergence:** [`results/figures/cka_divergence_curve.png`](results/figures/cka_divergence_curve.png) — Shows layer-by-layer similarity against ImageNet features, illustrating the divergence cliff.
- **Dynamic Routing Profiles:**
  - SVHN MobileNetV3: [`results/figures/transferguard_svhn_mobilenetv3_routing_profile.png`](results/figures/transferguard_svhn_mobilenetv3_routing_profile.png)
  - SVHN ShuffleNetV2: [`results/figures/transferguard_svhn_shufflenetv2_routing_profile.png`](results/figures/transferguard_svhn_shufflenetv2_routing_profile.png)
  - CIFAR-100: [`results/figures/transferguard_cifar100_mobilenetv3_routing_profile.png`](results/figures/transferguard_cifar100_mobilenetv3_routing_profile.png)
  - Flowers-102: [`results/figures/transferguard_flowers102_mobilenetv3_routing_profile.png`](results/figures/transferguard_flowers102_mobilenetv3_routing_profile.png)
- **Convergence Trajectories:**
  - SVHN MobileNetV3: [`results/figures/training_convergence_svhn_mobilenetv3.png`](results/figures/training_convergence_svhn_mobilenetv3.png)
  - SVHN ShuffleNetV2: [`results/figures/training_convergence_svhn_shufflenetv2.png`](results/figures/training_convergence_svhn_shufflenetv2.png)
  - CIFAR-100: [`results/figures/training_convergence_cifar100_mobilenetv3.png`](results/figures/training_convergence_cifar100_mobilenetv3.png)
  - Flowers-102: [`results/figures/training_convergence_flowers102_mobilenetv3.png`](results/figures/training_convergence_flowers102_mobilenetv3.png)

---

## 🚀 Quickstart Guide

### 1. Environment Setup
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

### 2. Fast CPU Sanity Check (Runs in < 15 seconds)
```bash
pytest tests/ -v
```
Verifies all 12 automated unit tests (CKA invariance, layer hooks, dynamic freezing, resumable training, and Google Drive auto-sync).

### 3. Google Colab GPU Execution
For full training on Tesla T4 GPU:
1. Open [`TransferGuard_Colab.ipynb`](./TransferGuard_Colab.ipynb) in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. All training commands support `--resume auto --drive-backup auto`: if your runtime disconnects, re-running resumes from the exact interrupted epoch.

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

# Linear Probing (Frozen Backbone - Demonstrates Negative Transfer)
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
Outputs `results/benchmark_summary.csv` and `results/benchmark_summary.json`.

### 6. Generate 300 DPI Publication Figures
```bash
python scripts/plot_results.py --output-dir results/figures
```

### 7. Interactive CLI Demo
```bash
python scripts/demo_live.py
```
Simulates real-time inference with colored ASCII meters showing stage gating scores and dynamic routing actions.

---

## 🧩 Extending TransferGuard (For Future Work)

TransferGuard is engineered modularly so you can easily plug in new vision backbones and datasets:

### Adding a New Backbone (e.g., EfficientNet, ResNet)
1. In `src/models/`, wrap your torchvision model and implement `forward_with_features()` with forward hooks at 5 architectural stage boundaries.
2. Register the wrapper in `src/models/adaptive_routing.py`.
3. Create a configuration in `configs/<dataset>_<backbone>.yaml`.
4. Run `python scripts/train_transferguard.py --config configs/<dataset>_<backbone>.yaml`.

### Adding a New Dataset
1. In `src/data/transforms.py`, add dataset-specific image sizes and normalization constants.
2. In `src/data/dataset_loader.py`, add torchvision dataset download and loader logic.
3. Configure the number of classes and paths in `configs/`.

---

## 🗂️ Repository Structure

```
.
├── TransferGuard_Colab.ipynb         # One-click GPU training notebook for Google Colab
├── README.md                         # Project overview, benchmarks, and quickstart
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
├── configs/                          # Experiment configuration YAMLs
│   ├── default.yaml
│   ├── svhn_mobilenetv3.yaml
│   ├── svhn_shufflenetv2.yaml
│   ├── cifar100_mobilenetv3.yaml
│   └── flowers102_mobilenetv3.yaml
├── docs/                             # In-depth architectural & reproducibility documentation
│   ├── REPRODUCIBILITY.md            # Hardware specs, seeds, and reproduction commands
│   ├── PROJECT_SPEC.md               # Theoretical foundation & mathematical formulations
│   ├── ARCHITECTURE_AND_DESIGN.md    # Detailed module design and dataflows
│   ├── EXPERIMENT_PLAN.md            # Systematic benchmarking plan & hypotheses
│   └── SYSTEM_WALKTHROUGH_AND_DATAFLOW.md # Technical implementation and tensor dataflows
├── src/                              # Core modular Python package
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
│   ├── plot_results.py               # Generates 300 DPI publication figures
│   └── demo_live.py                  # Interactive CLI inference demo
└── tests/                            # Unit tests and CPU sanity checks
    ├── test_cka.py
    ├── test_models.py
    └── test_pipeline_dryrun.py
```

---

## 📄 License
This project is licensed under the MIT License.
