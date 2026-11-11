"""
src/metrics/cka.py
Linear Centered Kernel Alignment (CKA) computation engine.
Measures functional representational similarity between hidden representations
extracted from neural networks invariant to orthogonal transformations and isotropic scaling.

Reference: Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019.
Authors: Arvind Yadav, Ghanishth Saini, Aakash Jagtap
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    """Centers Gram matrix K in-place or via matrix operations."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device, dtype=K.dtype) - (1.0 / n) * torch.ones(
        (n, n), device=K.device, dtype=K.dtype
    )
    return H @ K @ H


def _unbiased_hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Computes unbiased estimator of Hilbert-Schmidt Independence Criterion (HSIC)
    (Song et al., 2012; Kornblith et al., ICML 2019).
    """
    n = K.shape[0]
    if n < 4:
        # Fallback to standard HSIC for tiny mini-batches
        Kc = _center_gram(K)
        Lc = _center_gram(L)
        return torch.sum(Kc * Lc) / ((n - 1) ** 2)

    # Zero diagonals
    K_zero_diag = K.clone()
    K_zero_diag.fill_diagonal_(0.0)
    L_zero_diag = L.clone()
    L_zero_diag.fill_diagonal_(0.0)

    kl = torch.sum(K_zero_diag * L_zero_diag.t())
    k_sum = torch.sum(K_zero_diag)
    l_sum = torch.sum(L_zero_diag)
    ones = torch.ones((n, 1), device=K.device, dtype=K.dtype)

    term1 = kl
    term2 = (k_sum * l_sum) / ((n - 1) * (n - 2))
    term3 = (2.0 / (n - 2)) * torch.sum(
        (K_zero_diag @ ones) * (L_zero_diag @ ones)
    )

    hsic = (term1 + term2 - term3) / (n * (n - 3))
    return hsic


def linear_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    unbiased: bool = True,
) -> float:
    """
    Computes Linear Centered Kernel Alignment between activation matrices X and Y.
    
    Args:
        X: Tensor of shape [N, D1] (activations of representation 1).
        Y: Tensor of shape [N, D2] (activations of representation 2).
        unbiased: Whether to use the unbiased HSIC estimator.
        
    Returns:
        Scalar CKA score in range [0, 1].
    """
    if X.dim() > 2:
        X = X.view(X.shape[0], -1)
    if Y.dim() > 2:
        Y = Y.view(Y.shape[0], -1)

    X = X.float()
    Y = Y.float()

    # Center representations
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    if not unbiased:
        # Fast Frobenius inner product computation: ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
        dot_product = torch.norm(torch.matmul(Y.t(), X)) ** 2
        norm_x = torch.norm(torch.matmul(X.t(), X))
        norm_y = torch.norm(torch.matmul(Y.t(), Y))
        cka = dot_product / (norm_x * norm_y + 1e-12)
        return float(torch.clamp(cka, 0.0, 1.0).item())

    # Unbiased Gram matrix approach
    K = torch.matmul(X, X.t())
    L = torch.matmul(Y, Y.t())

    hsic_kl = _unbiased_hsic(K, L)
    hsic_kk = _unbiased_hsic(K, K)
    hsic_ll = _unbiased_hsic(L, L)

    denominator = torch.sqrt(torch.clamp(hsic_kk * hsic_ll, min=1e-12))
    cka = hsic_kl / denominator
    return float(torch.clamp(cka, 0.0, 1.0).item())


def rbf_kernel(X: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    """Computes RBF kernel matrix with median heuristic bandwidth."""
    dist_sq = torch.cdist(X, X) ** 2
    if sigma is None:
        # Median heuristic
        median_dist = torch.median(dist_sq[dist_sq > 0])
        sigma = torch.sqrt(median_dist / 2.0).item() if median_dist > 0 else 1.0
    gamma = 1.0 / (2.0 * (sigma ** 2) + 1e-12)
    return torch.exp(-gamma * dist_sq)


def kernel_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
) -> float:
    """
    Computes RBF Kernel Centered Kernel Alignment between activations X and Y.
    """
    if X.dim() > 2:
        X = X.view(X.shape[0], -1)
    if Y.dim() > 2:
        Y = Y.view(Y.shape[0], -1)

    X = X.float()
    Y = Y.float()

    K = rbf_kernel(X, sigma=sigma_x)
    L = rbf_kernel(Y, sigma=sigma_y)

    hsic_kl = _unbiased_hsic(K, L)
    hsic_kk = _unbiased_hsic(K, K)
    hsic_ll = _unbiased_hsic(L, L)

    denominator = torch.sqrt(torch.clamp(hsic_kk * hsic_ll, min=1e-12))
    cka = hsic_kl / denominator
    return float(torch.clamp(cka, 0.0, 1.0).item())


def compute_layer_cka_matrix(
    features_a: Dict[str, torch.Tensor],
    features_b: Dict[str, torch.Tensor],
    unbiased: bool = True,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Computes a full cross-layer CKA similarity matrix between two sets of layer representations.
    
    Returns:
        matrix: 2D numpy array [L_A, L_B]
        layers_a: list of layer names for model A
        layers_b: list of layer names for model B
    """
    layers_a = list(features_a.keys())
    layers_b = list(features_b.keys())

    matrix = np.zeros((len(layers_a), len(layers_b)), dtype=np.float32)

    for i, name_a in enumerate(layers_a):
        feat_a = features_a[name_a]
        for j, name_b in enumerate(layers_b):
            feat_b = features_b[name_b]
            matrix[i, j] = linear_cka(feat_a, feat_b, unbiased=unbiased)

    return matrix, layers_a, layers_b


class CenteredKernelAlignment:
    """Class wrapper for computing and accumulating CKA across mini-batches."""

    def __init__(self, unbiased: bool = True):
        self.unbiased = unbiased

    def compare_layers(
        self,
        features_src: Dict[str, torch.Tensor],
        features_tgt: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Computes diagonal layer-to-corresponding-layer CKA scores."""
        scores = {}
        for name in features_src.keys():
            if name in features_tgt:
                scores[name] = linear_cka(
                    features_src[name], features_tgt[name], unbiased=self.unbiased
                )
        return scores

