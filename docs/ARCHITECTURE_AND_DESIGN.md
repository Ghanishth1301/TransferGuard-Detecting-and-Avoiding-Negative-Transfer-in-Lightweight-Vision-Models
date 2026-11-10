# TransferGuard: System Architecture & Technical Design

This document details the architectural components, model hook mechanisms, representation extraction, CKA computation engine, TransferGuard gating networks, and adaptive layer routing algorithms.

---

## 1. High-Level Architecture Overview

```
+---------------------------------------------------------------------------------+
|                                 Input Batch                                     |
|                              (Target Domain X_T)                                |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
|                        Lightweight Vision Backbone                              |
|                   (MobileNetV3-Small / ShuffleNetV2)                            |
|                                                                                 |
|   +-------------------+    +-------------------+    +-----------------------+   |
|   |   Stage 1: Stem   |--->| Stage 2: Inverted |--->| Stage 3: Deep Blocks  |   |
|   |    Conv & Early   |    |  Residual Blocks  |    |  (High Receptive Fld) |   |
|   +-------------------+    +-------------------+    +-----------------------+   |
|             |                        |                          |               |
|             v (Activation Hook)      v (Activation Hook)        v (Hook)        |
+-------------|------------------------|--------------------------|---------------+
              |                        |                          |
              +------------------------+--------------------------+
                                       |
                                       v
        +--------------------------------------------------------------+
        |                 Representation Extraction Unit               |
        |             Global Pooling & Normalized Gram Matrices        |
        +--------------------------------------------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
    +-----------------------------+       +---------------------------------+
    |   Offline / Diagnostic:     |       |    Online Adaptive Execution:   |
    |   CKA Engine (Layer-wise)   |       |   TransferGuard Gating Network  |
    |                             |       |                                 |
    |  - Linear CKA Computation   |       |  - Lightweight MLP / Metric     |
    |  - Domain Alignment Heatmap |       |  - Computes Transfer Score S_l  |
    |  - Negative Transfer Alert  |       |  - Decides: Freeze/Tune/Bypass  |
    +-----------------------------+       +---------------------------------+
                   |                                       |
                   |                                       v
                   |                      +---------------------------------+
                   |                      |   Adaptive Controller / Router  |
                   |                      |                                 |
                   |                      | - Layer-wise Freeze Mask        |
                   |                      | - Dynamic Residual Scaling      |
                   |                      | - Gradient Routing              |
                   |                      +---------------------------------+
                   \_______________________________________/
                                       |
                                       v
        +--------------------------------------------------------------+
        |                 Classifier Head & Optimization               |
        |                  Loss: L_task + lambda * L_gate              |
        +--------------------------------------------------------------+
```

---

## 2. Core Modules Specification

### 2.1 Feature Hooks and Backbone Wrappers (`src/models/`)
PyTorch models in `torchvision.models` (MobileNetV3 and ShuffleNetV2) are wrapped to expose layer-level intermediate activations without modifying their internal forward signatures:
- **`MobileNetV3Wrapper`**: Extracts representations after:
  - Initial ConvBNActivation stem
  - Inverted Residual Blocks (Stages 1 through 6)
  - Final pointwise convolution block
- **`ShuffleNetV2Wrapper`**: Extracts representations after:
  - Stage 1 (Initial Conv + MaxPool)
  - Stage 2 (Shuffle blocks)
  - Stage 3 (Shuffle blocks)
  - Stage 4 (Shuffle blocks)
  - Final Conv5 block

Both wrappers provide forward hooks that collect activation tensors:
$$A_l = f_l(x) \in \mathbb{R}^{B \times C_l \times H_l \times W_l}$$
and spatially average them into compact vector representations:
$$z_l = \frac{1}{H_l W_l} \sum_{h=1}^{H_l} \sum_{w=1}^{W_l} A_l[:, :, h, w] \in \mathbb{R}^{B \times C_l}$$

### 2.2 Centered Kernel Alignment (CKA) Engine (`src/metrics/cka.py`)
The CKA engine computes both **Linear CKA** and **Kernel (RBF) CKA** efficiently in mini-batches:
1. Given activations $X \in \mathbb{R}^{N \times d_1}$ (from Source/Pre-trained) and $Y \in \mathbb{R}^{N \times d_2}$ (from Target):
2. Mean-center $X$ and $Y$ column-wise: $\tilde{X} = X - \mu_X$, $\tilde{Y} = Y - \mu_Y$.
3. Compute Gram matrices: $K = \tilde{X} \tilde{X}^T$, $L = \tilde{Y} \tilde{Y}^T$.
4. Unbiased HSIC estimator:
   $$\text{HSIC}(K, L) = \frac{1}{N(N-3)} \left( \text{tr}(\tilde{K} \tilde{L}) + \frac{\mathbf{1}^T \tilde{K} \mathbf{1} \mathbf{1}^T \tilde{L} \mathbf{1}}{(N-1)(N-2)} - \frac{2}{N-2} \mathbf{1}^T \tilde{K} \tilde{L} \mathbf{1} \right)$$
   where $\tilde{K}, \tilde{L}$ have their diagonals zeroed out.
5. In addition, an ultra-fast linear formulation using Frobenius inner product is implemented:
   $$\text{CKA}_{\text{linear}}(X, Y) = \frac{\| Y^T X \|_F^2}{\| X^T X \|_F \| Y^T Y \|_F}$$
   This allows real-time calculation even on CPU without computing $N \times N$ Gram matrices.

### 2.3 TransferGuard Gating Module (`src/models/transferguard_gate.py`)
The TransferGuard Gate operates at each backbone stage $l$:
- **Input:** Stage feature representation $z_l \in \mathbb{R}^C$.
- **Architecture:** 
  $$\text{Gate}_l(z_l) = \text{Sigmoid}\left( \mathbf{W}_2 \text{ReLU}(\mathbf{W}_1 z_l + \mathbf{b}_1) + b_2 \right) \in [0, 1]$$
  where $\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C}$ is a bottleneck reduction matrix (reduction ratio $r = 8$ or $16$) and $\mathbf{W}_2 \in \mathbb{R}^{1 \times \frac{C}{r}}$.
- **Parameter footprint:** Less than $0.5\%$ of total backbone parameters, preserving the lightweight nature of MobileNet/ShuffleNet.

### 2.4 Adaptive Layer Freezing & Routing Controller (`src/models/adaptive_routing.py`)
The routing controller handles layer updates dynamically:
1. **Dynamic Freezing:**
   - In early training epochs (epochs 1-3), the controller monitors the transferability score $S_l$ or validation transfer metrics.
   - If $S_l \ge \tau_{\text{freeze}}$ (e.g., $0.70$):
     $$\text{requires\_grad}(\theta_l) \leftarrow \text{False}$$
     Weights remain locked at pre-trained values, preventing gradient distortion.
2. **Residual Bypass Routing:**
   - For layers experiencing severe negative transfer ($S_l < \tau_{\text{bypass}}$, e.g. $0.35$):
     $$y_{l+1} = (1 - \alpha_l) \cdot \text{Block}_l(y_l) + \alpha_l \cdot \text{Shortcut}(y_l)$$
     where $\alpha_l = 1 - S_l$. This softens or bypasses the conflicting pre-trained transformation, allowing the target task to converge without obstacle.

---

## 3. Training & Optimization Pipelines

### 3.1 Training Regimes
1. **`Scratch` Baseline:** 
   - Backbone randomly initialized using Kaiming normal initialization.
   - All layers trained with CrossEntropyLoss and cosine annealing learning rate.
2. **`Standard Fine-Tuning` Baseline:**
   - Backbone initialized with official PyTorch ImageNet-1K pre-trained weights.
   - Classifier head replaced with target num_classes.
   - All layers updated with standard learning rate $\eta$.
3. **`Linear Probing` Baseline:**
   - Backbone frozen completely ($\text{requires\_grad} = \text{False}$).
   - Only the new linear classifier head is trained.
4. **`TransferGuard Adaptive Routing` (Proposed):**
   - Pre-trained backbone with TransferGuard controller.
   - Dynamic stage-wise freezing and adaptive routing active.
   - Objective:
     $$\mathcal{L} = \mathcal{L}_{\text{task}}(y, \hat{y}) + \lambda \mathcal{L}_{\text{reg}}(\mathbf{S})$$

---

## 4. Hardware Optimization Strategy (CPU & Colab GPU)
- **Local CPU Mode:**
  - Fast dev-run flag (`--debug`, `--subset 0.05`, `--batch-size 32`, `--num-workers 2`) allows running full unit tests, metric calculations, and sanity checks locally on Intel Core i5 in under 2 minutes.
- **Google Colab Mode:**
  - One-click standalone notebook (`TransferGuard_Colab.ipynb`) clones or downloads the repo, loads GPU (NVIDIA T4 / V100), executes full training runs, downloads checkpoints, and plots publication-quality figures directly into an output directory.

