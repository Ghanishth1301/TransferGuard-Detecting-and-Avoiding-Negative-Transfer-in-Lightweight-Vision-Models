# TransferGuard: Detecting and Avoiding Negative Transfer in Lightweight Vision Models
## Comprehensive Project Specification

**Course:** Computer Vision (CV) Project  
**Authors:**
- **Ghanishth Saini** (Roll: 24121)
- **Arvind Yadav** (Roll: 24059)
- **Aakash Jagtap** (Roll: 24157)

---

## 1. Executive Summary & Problem Formulation

### 1.1 Motivation
Transfer learning from large-scale natural image datasets (e.g., ImageNet-1K with 1.28M images across 1,000 categories) is the standard paradigm for modern computer vision tasks. Conventional wisdom holds that pre-training provides rich, generic visual representations (Gabor-like edge detectors, textures, motifs) that accelerate convergence and boost downstream classification accuracy.

However, transfer learning rests on a fundamental assumption: **the source domain distribution $\mathcal{P}_S(X, Y)$ and target domain distribution $\mathcal{P}_T(X, Y)$ share compatible feature manifolds.** 

When this assumption breaks down—most prominently in domain mismatch scenarios such as transferring from ImageNet (natural objects, animals, vehicles) to Street View House Numbers (SVHN; cropped digits under severe distortion, illumination variation, and synthetic glyphs)—fine-tuning pre-trained weights often leads to **Negative Transfer**. In these situations, fine-tuning performs demonstrably worse than training an identical network from scratch with random initialization.

### 1.2 The Specific Vulnerability of Lightweight Vision Models
While over-parameterized models (e.g., ViT-Large, ResNet-101) possess surplus parameter capacity to overwrite conflicting source features without sacrificing optimization dynamics, **lightweight vision models** (e.g., **MobileNetV3**, **ShuffleNetV2**) operate under severe capacity constraints:
- Compact parameter budgets ($1.5\text{M} - 5.5\text{M}$ parameters)
- Narrow bottleneck channels and depthwise-separable convolutions
- Constrained gradient bandwidth and fragile optimization landscapes

In lightweight backbones, deep feature representations learned on natural objects actively impede learning structured numerical glyphs. Forcing end-to-end backpropagation through pre-trained deep layers distorts early generic feature maps, leading to catastrophic degradation.

### 1.3 Project Core Objectives
1. **Identify and Characterize Negative Transfer:** Empirically quantify negative transfer across diverse domain mismatches in lightweight architectures (MobileNetV3-Small, MobileNetV3-Large, ShuffleNetV2).
2. **Formulate TransferGuard Predictive Metric:** Utilize Centered Kernel Alignment (CKA) and feature divergence metrics across layers to measure layer-wise representational divergence prior to or during fine-tuning.
3. **Adaptive Layer Routing & Dynamic Freezing:** Develop the `TransferGuard` adaptive module that automatically identifies harmful layers and executes selective freezing or residual bypass routing, transforming negative transfer into positive transfer gain.

---

## 2. Mathematical Framework & Metrics

### 2.1 Transfer Gain ($TG$)
The fundamental metric for detecting negative transfer is **Transfer Gain ($TG$)**:

$$TG = \text{Accuracy}_{\text{Transfer}} - \text{Accuracy}_{\text{Scratch}}$$

Where:
- $\text{Accuracy}_{\text{Transfer}}$: Top-1 test accuracy achieved by fine-tuning the ImageNet pre-trained backbone on the target dataset.
- $\text{Accuracy}_{\text{Scratch}}$: Top-1 test accuracy achieved by training the exact same architecture initialized with random Gaussian/He weights on the target dataset.

**Interpretation:**
- $TG > 0$: **Positive Transfer** (Source knowledge assists target domain learning).
- $TG \approx 0$: **Neutral Transfer** (Source knowledge neither helps nor hurts).
- $TG < 0$: **Negative Transfer** (Pre-trained weights actively degrade target task performance).

### 2.2 Centered Kernel Alignment (CKA)
To understand *where* in the deep architecture representations diverge between source and target domains, we employ **Linear Centered Kernel Alignment (CKA)** (Kornblith et al., ICML 2019). CKA is invariant to orthogonal transformations and isotropic scaling, making it the gold standard for comparing hidden representations across layers and datasets.

Let $X \in \mathbb{R}^{m \times p}$ and $Y \in \mathbb{R}^{m \times q}$ be centered activation matrices from a given layer for $m$ target samples passing through:
1. The ImageNet pre-trained feature extractor ($X$)
2. The target fine-tuned or scratch-trained feature extractor ($Y$)

The Gram matrices are $K = X X^T$ and $L = Y Y^T$. The Hilbert-Schmidt Independence Criterion (HSIC) is defined as:

$$\text{HSIC}(K, L) = \frac{1}{(m-1)^2} \text{tr}(K H L H)$$

Where $H = I_m - \frac{1}{m} \mathbf{1}\mathbf{1}^T$ is the centering matrix. Linear CKA is then formulated as:

$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}} \in [0, 1]$$

**Diagnostic Utility:**
- A high CKA in shallow layers (layers 1-3) confirms low-level features (edges, textures) are transferable.
- A steep drop in CKA in deep layers (layers 8-15) signals severe representational incompatibility (the root cause of negative transfer).

### 2.3 TransferGuard Gating Score ($S_l$)
For each building block or stage $l \in \{1, 2, \dots, L\}$, TransferGuard computes a transferability index $S_l \in [0, 1]$:

$$S_l = \sigma \left( \mathbf{w}_l^T \phi(z_l) + b_l \right)$$

Where $\phi(z_l)$ is a lightweight projection (global average pooling + layer normalization) of target features $z_l$, or derived from layer-wise CKA alignment with canonical source anchor statistics.

### 2.4 Adaptive Routing Decision
Based on $S_l$ and an adaptive threshold $\tau_l$:
- **Case 1 ($S_l \ge \tau_{\text{high}}$): Frozen Transfer.** Layer features are highly compatible. Freeze weights ($\nabla_{\theta_l} = 0$) to preserve generic representations and save compute.
- **Case 2 ($\tau_{\text{low}} \le S_l < \tau_{\text{high}}$): Plastic Fine-Tuning.** Layer is moderately compatible. Train with regularized learning rate $\eta_l = \eta \cdot S_l$.
- **Case 3 ($S_l < \tau_{\text{low}}$): Harmful Transfer Bypass.** Layer induces severe negative transfer. The routing mechanism bypasses pre-trained weights via an identity/projection residual shortcut, or resets layer weights to random initialization, shielding downstream representations from harmful source interference.

---

## 3. Dataset Specifications

| Dataset | Split | Images | Classes | Image Resolution | Domain Type | Expected Transfer Regime |
|---|---|---|---|---|---|---|
| **ImageNet-1K (Source)** | Pre-trained | 1,281,167 | 1000 | $224 \times 224$ | Natural objects, animals, everyday items | Source Domain |
| **SVHN (Target)** | Train / Test | 73,257 / 26,032 | 10 | $32 \times 32 \to 224 \times 224$ | Real-world house numbers, cropped digits | **Severe Negative Transfer ($TG < 0$)** |
| **CIFAR-100 (Target)** | Train / Test | 50,000 / 10,000 | 100 | $32 \times 32 \to 224 \times 224$ | Natural objects, animals, vehicles | **Mild / Positive Transfer ($TG > 0$)** |
| **Oxford Flowers-102 (Target)** | Train / Val / Test | 1,020 / 1,020 / 6,149 | 102 | Variable $\to 224 \times 224$ | Fine-grained botanical flowers | **High Transfer Gain ($TG \gg 0$)** |

---

## 4. Target Architectures

### 4.1 MobileNetV3 (Small & Large)
- **Key Mechanics:** Inverted Residual Blocks with Linear Bottlenecks, Depthwise Separable Convolutions, Squeeze-and-Excitation (SE) attention, Hard-Swish activations.
- **Small Variant:** $\sim 2.5\text{M}$ parameters, $0.06$ GFLOPs.
- **Large Variant:** $\sim 5.4\text{M}$ parameters, $0.22$ GFLOPs.

### 4.2 ShuffleNetV2 ($1.0\times$)
- **Key Mechanics:** Channel Split, Depthwise Convolutions, Channel Shuffle for cross-group information mixing. Explicitly designed to minimize Memory Access Cost (MAC) on mobile/edge hardware.
- **Parameters:** $\sim 2.3\text{M}$ parameters, $0.15$ GFLOPs.

---

## 5. Deliverables & Expected Outcomes
1. **End-to-End Reproducible Repository:** Modular PyTorch code supporting CPU dry-runs, automated testing, and full Google Colab execution with GPU acceleration.
2. **Empirical Verification:** Demonstration of negative transfer on SVHN ($TG < 0$) with standard fine-tuning, and recovery ($TG > 0$) using TransferGuard adaptive routing.
3. **Visual Diagnostics:** Automated generation of publication-quality CKA heatmaps, Transfer Gain comparisons, layer freezing distributions, and training loss/accuracy trajectories.
4. **Presentation & Defense Readiness:** Complete presentation guide, speaker scripts, and technical Q&A preparation for evaluation.

