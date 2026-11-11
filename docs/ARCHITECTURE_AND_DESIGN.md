# TransferGuard: System Architecture & Technical Design

> **Authors:** Arvind Yadav, Ghanishth Saini, Aakash Jagtap  
> **Repository:** https://github.com/arvindyadav186/TransferGuard-Detecting-and-Avoiding-Negative-Transfer-in-Lightweight-Vision-Models.git  

This document details the architectural components, model hook mechanisms, representation extraction, CKA computation engine, TransferGuard gating networks, and adaptive layer routing algorithms as implemented in the codebase.

---

## 1. High-Level Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 Input Batch                                     │
│                              (Target Domain X_T)                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Lightweight Vision Backbone                              │
│                   (MobileNetV3-Small / ShuffleNetV2)                            │
│                                                                                 │
│   ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────────┐   │
│   │   Stage 1: Stem   │───►│ Stage 2: Inverted │───►│ Stage 5: Deep Blocks  │   │
│   │    Conv & Early   │    │  Residual Blocks  │    │  (Class Abstraction)  │   │
│   └───────────────────┘    └───────────────────┘    └───────────────────────┘   │
│             │                        │                          │               │
│             ▼ (Forward Hook)         ▼ (Forward Hook)           ▼ (Hook)        │
└─────────────│────────────────────────│──────────────────────────│───────────────┘
              │                        │                          │
              └────────────────────────┼──────────────────────────┘
                                       │
                                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                 Representation Extraction Unit               │
        │             Global Average Pooling (GAP) -> [B, C_l]         │
        └──────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
    ┌─────────────────────────────┐       ┌─────────────────────────────────┐
    │   Diagnostic Analysis:      │       │    Online Adaptive Execution:   │
    │   CKA Engine (Layer-wise)   │       │   TransferGuard Gating Network  │
    │                             │       │                                 │
    │  - Linear CKA Computation   │       │  - Micro-MLP per Stage          │
    │  - Domain Alignment Curves  │       │  - Computes Transfer Score S_l  │
    │  - Negative Transfer Alert  │       │  - Bimodal Entropy Loss         │
    └─────────────────────────────┘       └─────────────────────────────────┘
                   │                                       │
                   │                                       ▼
                   │                      ┌─────────────────────────────────┐
                   │                      │   Adaptive Controller / Router  │
                   │                      │                                 │
                   │                      │ - EMA Score Accumulator         │
                   │                      │ - Dynamic Stage Freezing        │
                   │                      │ - Gradient Gating (Bypass/Tune) │
                   │                      └─────────────────────────────────┘
                   \───────────────────────────────────────/
                                       │
                                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                 Classifier Head & Optimization               │
        │                  Loss = L_task + λ * L_entropy               │
        └──────────────────────────────────────────────────────────────┘
```

---

## 2. Core Modules Specification

### 2.1 Feature Hooks and Backbone Wrappers (`src/models/`)
PyTorch models from `torchvision.models` (MobileNetV3-Small and ShuffleNetV2 1.0x) are wrapped to expose layer-level intermediate activations without modifying their internal forward signatures:

- **`MobileNetV3Wrapper` (`src/models/mobilenet_v3.py`):**  
  Extracts representations at 5 progressive architectural stage boundaries across the 13 inverted residual blocks:
  - `stem`: Initial Conv3x3 + BatchNorm + Hardswish (`features[0]`)
  - `stage_early`: Early inverted residual block (`features[2]`)
  - `stage_mid_1`: Mid-1 inverted residual block (`features[4]`)
  - `stage_mid_2`: Mid-2 inverted residual block (`features[6]`)
  - `stage_deep`: Final deep inverted residual block (`features[12]`)

- **`ShuffleNetV2Wrapper` (`src/models/shufflenet_v2.py`):**  
  Extracts representations at 5 stage boundaries across channel-split shuffle units:
  - `stem`: Initial Conv1 (3x3 conv) & MaxPool
  - `stage_2`: Stage 2 channel-split shuffle blocks
  - `stage_3`: Stage 3 channel-split shuffle blocks
  - `stage_4`: Stage 4 channel-split shuffle blocks
  - `stage_deep`: Final Conv5 pointwise convolution block

Both wrappers provide forward hooks that collect activation tensors:

$$
A_l = f_l(x) \in \mathbb{R}^{B \times C_l \times H_l \times W_l}
$$

and spatially average them into compact vector representations:

$$
z_l = \frac{1}{H_l W_l} \sum_{h=1}^{H_l} \sum_{w=1}^{W_l} A_l[:, :, h, w] \in \mathbb{R}^{B \times C_l}
$$

---

### 2.2 Centered Kernel Alignment (CKA) Engine (`src/metrics/cka.py`)
The CKA engine computes functional representational similarity between hidden feature representations across layers and models:

1. Given activation matrices $X \in \mathbb{R}^{N \times d_1}$ (from Source/Pre-trained) and $Y \in \mathbb{R}^{N \times d_2}$ (from Target).
2. Mean-center $X$ and $Y$ column-wise: $\tilde{X} = X - \mu_X$, $\tilde{Y} = Y - \mu_Y$.
3. Compute Gram matrices: $K = \tilde{X} \tilde{X}^T$, $L = \tilde{Y} \tilde{Y}^T$.

#### Unbiased HSIC Estimator
Given centered Gram matrices $\tilde{K}$ and $\tilde{L}$ with diagonals zeroed out:

$$
\text{HSIC}(K, L) = \frac{1}{N(N-3)} \left( \text{tr}(\tilde{K} \tilde{L}) + \frac{\mathbf{1}^T \tilde{K} \mathbf{1} \mathbf{1}^T \tilde{L} \mathbf{1}}{(N-1)(N-2)} - \frac{2}{N-2} \mathbf{1}^T \tilde{K} \tilde{L} \mathbf{1} \right)
$$

#### Linear CKA Formulation
Normalized similarity index bounded in $[0, 1]$:

$$
\text{CKA}_{\text{linear}}(X, Y) = \frac{\| Y^T X \|_F^2}{\| X^T X \|_F \| Y^T Y \|_F}
$$

This allows real-time calculation even on CPU without computing full $N \times N$ Gram matrices.

---

### 2.3 TransferGuard Gating Module (`src/models/transferguard_gate.py`)
Each stage possesses a dedicated micro-gate parameterized as an MLP with LayerNorm and non-linearities:

$$
\text{Gate}_l(z_l) = \text{Sigmoid}\left( \mathbf{W}_2 \cdot \text{ReLU}\left( \text{LayerNorm}(\mathbf{W}_1 z_l + \mathbf{b}_1) \right) + b_2 \right) \in [0, 1]
$$

where $\mathbf{W}_1 \in \mathbb{R}^{\frac{C_l}{r} \times C_l}$ is a bottleneck projection matrix with reduction ratio $r = 8$ and $\mathbf{W}_2 \in \mathbb{R}^{1 \times \frac{C_l}{r}}$.

- **Input:** Spatially pooled stage representation $z_l \in \mathbb{R}^{C_l}$.
- **Parameter Footprint:** Under $0.5\%$ of total backbone parameters, preserving edge deployment feasibility.
- **Bimodal Entropy Regularization:**  
  To prevent micro-gates from stalling at neutral values ($0.5$), the controller optimizes an entropy loss:

$$
\mathcal{H}(p) = - \left[ p \log(p) + (1 - p) \log(1 - p) \right]
$$

  Minimizing entropy drives scores decisively toward $1.0$ (freeze) or $0.0$ (bypass). Cast to `float32` for numerical stability under AMP.

---

### 2.4 Adaptive Layer Freezing & Routing Controller (`src/models/adaptive_routing.py`)
The routing controller handles layer updates dynamically:

1. **Exponential Moving Average (EMA) Score Tracking:**
   Running transferability scores are maintained across mini-batches:

$$
S_{\text{running}} \leftarrow \alpha \cdot S_{\text{running}} + (1 - \alpha) \cdot S_{\text{batch}}
$$

2. **Threshold Decisions:**
   - **Freeze Action ($S_l \ge \tau_{\text{freeze}}$, default: $0.65$ / $0.70$):**  
     Sets $\text{requires\_grad}(\theta_l) \leftarrow \text{False}$. Pre-trained weights are locked, preserving generic edge/texture representations and eliminating backpropagation overhead.
   - **Bypass / Adapt Action ($S_l \le \tau_{\text{bypass}}$, default: $0.35$ / $0.40$):**  
     Identifies harmful mismatch in deep layers, bypassing conflicting source weights and allowing the target task to optimize unhindered.
   - **Tune Action ($\tau_{\text{bypass}} < S_l < \tau_{\text{freeze}}$):**  
     Standard plastic fine-tuning with full gradient updates.

---

## 3. Training & Optimization Pipelines

### 3.1 Training Regimes
1. **`Scratch` Baseline:**  
   Backbone randomly initialized using Kaiming normal initialization. All layers trained with CrossEntropyLoss and cosine annealing learning rate.
2. **`Standard Fine-Tuning` Baseline:**  
   Backbone initialized with official PyTorch ImageNet-1K pre-trained weights. Classifier replaced with target num_classes. All layers updated with standard learning rate.
3. **`Linear Probing` Baseline:**  
   Backbone completely frozen ($\text{requires\_grad} = \text{False}$). Only the linear classification head is trained.
4. **`TransferGuard Adaptive Routing` (Proposed):**  
   Pre-trained backbone with TransferGuard controller. Dynamic stage-wise freezing and adaptive routing active:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(y, \hat{y}) + \lambda_{\text{entropy}} \mathcal{H}(\mathbf{S})
$$

---

## 4. Hardware Optimization & Cloud Persistence

- **Automatic Mixed Precision (AMP):**  
  Uses `torch.cuda.amp.autocast()` and `GradScaler` to accelerate training throughput on Tensor Core GPUs (Tesla T4, RTX 3080/4090).
- **Zero-Loss Checkpoint Persistence:**  
  Saves `latest_model.pth` and `best_model.pth` every epoch. Automatic bidirectional mirroring with Google Drive (`--drive-backup auto`) ensures that if a cloud session disconnects, training resumes seamlessly with zero lost compute.
- **Fast CPU Dry-Run Verification:**  
  Unit tests (`pytest tests/ -v`) execute in under $15$ seconds on standard CPUs using lightweight synthetic batches.
