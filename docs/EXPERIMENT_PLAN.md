# TransferGuard: Experimental Plan & Benchmarking Protocol

This document outlines the systematic experiment matrix, dataset pre-processing, baseline comparisons, evaluation metrics, ablation studies, and execution schedule.

---

## 1. Experimental Matrix

To thoroughly evaluate the TransferGuard mechanism against negative transfer, the experiments span **3 target datasets** $\times$ **2 lightweight vision backbones** $\times$ **4 training methods**:

### 1.1 Datasets
1. **SVHN (Street View House Numbers):**
   - *Domain mismatch:* Natural objects (ImageNet) $\to$ Digit recognition under ambient clutter.
   - *Expected behavior:* **Severe negative transfer** ($TG < 0$) in standard fine-tuning due to conflicting deep feature priors.
2. **CIFAR-100:**
   - *Domain mismatch:* Coarse natural objects $\to$ Fine-grained 100-class natural objects.
   - *Expected behavior:* **Moderate/Positive transfer** ($TG > 0$).
3. **Oxford Flowers-102:**
   - *Domain mismatch:* Broad natural domain $\to$ Specific botanical domain (small training set: $10$ images per class).
   - *Expected behavior:* **Strong positive transfer** ($TG \gg 0$) where pre-training is crucial due to sample scarcity.

### 1.2 Backbones
1. **MobileNetV3-Small** (Torchvision: `mobilenet_v3_small(weights='DEFAULT')`)
2. **ShuffleNetV2 1.0x** (Torchvision: `shufflenet_v2_x1_0(weights='DEFAULT')`)

### 1.3 Methods for Comparison
1. **Train from Scratch (`scratch`):** Random initialization, no source knowledge.
2. **Standard Fine-Tuning (`finetune_full`):** Standard end-to-end backpropagation across all layers.
3. **Linear Probing (`linear_probe`):** Frozen backbone, only classification head trained.
4. **TransferGuard (`transferguard_adaptive`):** Proposed dynamic CKA-guided layer-wise gating and selective freezing.

---

## 2. Evaluation Protocol & Metrics

### 2.1 Quantitative Metrics
1. **Top-1 & Top-5 Test Accuracy (%):**
   $$\text{Acc}_1 = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}(\hat{y}_i = y_i)$$
2. **Transfer Gain ($TG$):**
   $$TG = \text{Acc}_{\text{Method}} - \text{Acc}_{\text{Scratch}}$$
   A negative value ($TG < 0$) signifies Negative Transfer. The goal of TransferGuard is to ensure $TG \ge 0$ across all regimes.
3. **Layer-wise CKA Alignment ($CKA_l$):**
   Pairwise similarity of layer $l$ representations between ImageNet pre-trained weights and target domain representations.
4. **Parameter Update Efficiency:**
   Percentage of backbone parameters frozen vs fine-tuned, and resulting speedup in backward pass.

---

## 3. Step-by-Step Experiment Execution

### Phase 1: Environment Setup & Sanity Verification (Local CPU)
- Create Python virtual environment (`.venv`).
- Install PyTorch, Torchvision, Scipy, Matplotlib, Seaborn, PyYAML, Tqdm.
- Run unit test suite:
  - CKA calculation unit tests (verifying invariance to orthogonal rotation and scaling).
  - Backbone wrapper hook verification for MobileNetV3 and ShuffleNetV2.
  - End-to-end dry-run with synthetic/mini batches to ensure 0 errors.

### Phase 2: Layer-wise CKA Diagnostic Analysis
- Compute layer-by-layer CKA between pre-trained ImageNet features and target datasets (SVHN, CIFAR-100, Flowers-102).
- Generate and save CKA heatmaps (`results/figures/cka_heatmap_*.png`).
- Validate the hypothesis: **SVHN exhibits low CKA scores in deep layers, predicting negative transfer.**

### Phase 3: Baseline Training (Google Colab GPU / Cloud)
- Train MobileNetV3-Small and ShuffleNetV2 from scratch on SVHN, CIFAR-100, and Flowers-102.
- Fine-tune pre-trained ImageNet backbones on the same datasets.
- Train linear probing baselines.
- Record validation curves, final test accuracies, and calculate Transfer Gain ($TG$).

### Phase 4: TransferGuard Implementation & Adaptive Freezing
- Train with TransferGuard dynamic layer gating on SVHN and CIFAR-100.
- Verify that TransferGuard identifies harmful deep layers in SVHN and automatically locks/bypasses them.
- Compare final accuracy against standard fine-tuning and scratch.

### Phase 5: Ablation Studies & Publication Assets
- **Ablation 1:** Freeze threshold sensitivity analysis ($\tau \in [0.4, 0.6, 0.8]$).
- **Ablation 2:** Compare TransferGuard adaptive routing vs fixed heuristic (freeze first $k$ layers).
- Generate publication figures:
  - Bar chart: Transfer Gain across datasets and methods (`transfer_gain_comparison.png`).
  - Line plot: Layer-wise CKA curves showing the "divergence cliff" (`cka_divergence_curve.png`).
  - Dynamic routing profiles (`transferguard_*_routing_profile.png`).
  - Convergence plot: Loss and accuracy curves over training epochs (`training_convergence_*.png`).

