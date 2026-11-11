# TransferGuard: System Architecture, Dataflows & Implementation Guide

> **Authors:** Arvind Yadav, Ghanishth Saini, Aakash Jagtap  
> **Repository:** https://github.com/arvindyadav186/TransferGuard-Detecting-and-Avoiding-Negative-Transfer-in-Lightweight-Vision-Models.git  

This document provides a technical walkthrough of the **TransferGuard** framework. It traces tensor dataflows, module hook mechanisms, dynamic gating logic, and provides a clear reference for developers looking to inspect or extend the codebase with new backbones and datasets.

---

## 1. High-Level Architecture & Concept

### The Problem: Negative Transfer in Lightweight Models
When fine-tuning ImageNet pre-trained vision models on target domains with substantial distribution shifts (such as Street View House Numbers, SVHN), standard transfer learning frequently causes **Negative Transfer** ($TG < 0$).
- **Early layers** encode generic, reusable visual primitives (Gabor-like edge filters, simple color contrasts, corners).
- **Deep layers** encode high-level semantic representations tightly coupled to natural objects (e.g., animal fur, dog faces, wheel spokes).
- Forcing end-to-end backpropagation through mismatched deep layers induces gradient interference that degrades the early generic filters, resulting in worse validation accuracy than training randomly initialized weights from scratch.

### The Solution: TransferGuard Adaptive Routing
TransferGuard introduces a non-invasive layer-wise gating mechanism:
1. **Dynamic Representation Hooks:** Extracts intermediate stage feature activations during forward passes without altering backbone architectures.
2. **Transferability Scoring ($S_l \in [0, 1]$):** Micro-MLP gating units evaluate feature compatibility for each stage.
3. **Selective Dynamic Freezing:**
   - If $S_l \ge \tau_{\text{freeze}}$ (e.g., $0.70$): Lock pre-trained weights to preserve transferable features and save compute.
   - If $S_l \le \tau_{\text{bypass}}$ (e.g., $0.35$): Bypass or isolate mismatched deep features to prevent destructive gradient interference.
   - Otherwise: Continue plastic fine-tuning with full gradient updates.

---

## 2. End-to-End Dataflow & Tensor Shapes

Tracing a mini-batch of images through the framework:

```
[ Input Images ] 
      │  Raw images from dataset (e.g., SVHN 32x32)
      ▼
src/data/transforms.py 
      │  Resize to 224x224, augment, normalize with ImageNet stats
      │  Tensor: [Batch_Size, 3, 224, 224]
      ▼
src/data/dataset_loader.py 
      │  Batching, pinned memory, persistent DataLoader workers
      ▼
src/models/mobilenet_v3.py (or shufflenet_v2.py)
      │  Backbone forward pass with PyTorch forward hooks registered at 5 stage boundaries:
      │    - Stem:        [Batch_Size, C_stem, H/2, W/2]  ──► GlobalPool ──► [Batch_Size, C_stem]
      │    - Stage Early: [Batch_Size, C_1, H/4, W/4]      ──► GlobalPool ──► [Batch_Size, C_1]
      │    - Stage Mid-1: [Batch_Size, C_2, H/8, W/8]      ──► GlobalPool ──► [Batch_Size, C_2]
      │    - Stage Mid-2: [Batch_Size, C_3, H/16, W/16]    ──► GlobalPool ──► [Batch_Size, C_3]
      │    - Stage Deep:  [Batch_Size, C_4, H/32, W/32]    ──► GlobalPool ──► [Batch_Size, C_4]
      ▼
src/models/transferguard_gate.py
      │  Each stage vector passes through a lightweight MLP:
      │  Linear(C_l, C_l // 8) ──► LayerNorm ──► ReLU ──► Linear(C_l // 8, 1) ──► Sigmoid
      │  Yields stage transferability score: S_l ∈ [0, 1]
      ▼
src/models/adaptive_routing.py
      │  Maintains Exponential Moving Average (EMA) of stage scores across batches
      │  At epoch boundaries, executes threshold logic:
      │    - S_l ≥ 0.70  ──► FREEZE  (param.requires_grad = False)
      │    - S_l ≤ 0.35  ──► BYPASS  (disable / reset conflicting weights)
      │    - otherwise   ──► TUNE    (standard backpropagation)
      ▼
src/training/loss.py
      │  Composite loss function:
      │  Total Loss = Task CrossEntropy + λ_entropy * Gate Entropy Loss
      ▼
src/training/trainer.py
      │  Backpropagation with PyTorch AMP (Automatic Mixed Precision via torch.cuda.amp)
      │  Fault-tolerant checkpoint persistence (mirrored to Google Drive if configured)
```

---

## 3. Core Component Architecture

### 3.1 Feature Extraction Hooks (`src/models/mobilenet_v3.py`, `shufflenet_v2.py`)
Rather than rewriting vision backbones from scratch, TransferGuard wraps official `torchvision.models` instances and registers standard PyTorch forward hooks (`register_forward_hook`).
When an image batch passes through the backbone, the forward hook captures activations at stage boundaries, computes spatial global average pooling along `dim=[2, 3]`, and populates a feature dictionary:
```python
def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    self._hook_features.clear()
    logits = self.backbone(x)
    features = {k: v for k, v in self._hook_features.items()}
    return logits, features
```

### 3.2 Micro-Gating Network (`src/models/transferguard_gate.py`)
Each stage possesses a separate micro-gate parameterized by an MLP with a reduction ratio (default: $8$). To ensure the gates make decisive binary choices rather than drifting around neutral values ($0.5$), an entropy minimization penalty is applied:
$$\mathcal{H}(p) = - [p \log(p) + (1 - p) \log(1 - p)]$$
Minimizing this entropy pushes the output toward $1.0$ (safe to freeze) or $0.0$ (harmful, bypass/tune).

### 3.3 Representation Alignment via Linear CKA (`src/metrics/cka.py`)
Centered Kernel Alignment (Kornblith et al., ICML 2019) evaluates functional similarity between activation matrices $X \in \mathbb{R}^{n \times d_1}$ and $Y \in \mathbb{R}^{n \times d_2}$:
$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$
where $K = XX^T$ and $L = YY^T$ are Gram matrices centered using $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$. CKA is invariant to orthogonal rotations and isotropic scaling, providing an accurate mathematical indicator of layer-wise domain divergence.

---

## 4. Module Reference Quick-Lookup

| Component | Key Class / Function | Primary File | Responsibilities |
|---|---|---|---|
| **MobileNetV3 Wrapper** | `MobileNetV3Wrapper` | `src/models/mobilenet_v3.py` | Stage grouping, hook extraction, selective stage freezing. |
| **ShuffleNetV2 Wrapper** | `ShuffleNetV2Wrapper` | `src/models/shufflenet_v2.py` | Channel-split stage hooks, feature pooling, stage freezing. |
| **Micro-Gate Unit** | `TransferGuardGate` | `src/models/transferguard_gate.py` | MLP mapping $z_l \to S_l \in [0, 1]$. |
| **Gating Controller** | `LayerGatingController` | `src/models/transferguard_gate.py` | Coordinates all stage gates; computes bimodal entropy loss. |
| **Adaptive Router** | `AdaptiveRoutingModel` | `src/models/adaptive_routing.py` | EMA score tracking; updates backbone `requires_grad` flags. |
| **CKA Metric Engine** | `linear_cka` | `src/metrics/cka.py` | Unbiased HSIC Gram matrix centering and representation similarity. |
| **Transfer Gain Metric** | `compute_transfer_gain` | `src/metrics/transfer_gain.py` | $TG = Acc_{\text{transfer}} - Acc_{\text{scratch}}$ calculation and diagnostics. |
| **Unified Trainer** | `Trainer` | `src/training/trainer.py` | AMP training loop, cosine annealing, validation, auto-resume. |
| **Composite Loss** | `TransferGuardLoss` | `src/training/loss.py` | Task Cross-Entropy + Gate Entropy Regularization. |
| **Data Pipelines** | `get_dataloader` | `src/data/dataset_loader.py` | SVHN, CIFAR-100, and Flowers-102 dataset loading. |

---

## 5. Guide: Extending TransferGuard with a New Model

To integrate a new vision backbone (e.g., EfficientNet-B0 or ResNet-18) in the future:

1. **Create a Wrapper in `src/models/`:**
   Implement `forward_with_features()` and register hooks at 5 architectural stage boundaries:
   ```python
   class EfficientNetWrapper(nn.Module):
       def __init__(self, pretrained=True, num_classes=10):
           super().__init__()
           self.backbone = torchvision.models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
           # Group layers into stem, stage_early, stage_mid_1, stage_mid_2, stage_deep
           ...
   ```
2. **Add to `src/models/adaptive_routing.py`:**
   Include the new wrapper type in the `AdaptiveRoutingModel` type annotations and stage-channel probe.
3. **Add Config in `configs/`:**
   Define `configs/<dataset>_<newmodel>.yaml` with the model name, hyperparameters, and batch sizes.
4. **Train & Evaluate:**
   Run `scripts/train_transferguard.py --config configs/<dataset>_<newmodel>.yaml`. TransferGuard will automatically probe stage dimensions, instantiate micro-gates, and execute dynamic layer routing.
