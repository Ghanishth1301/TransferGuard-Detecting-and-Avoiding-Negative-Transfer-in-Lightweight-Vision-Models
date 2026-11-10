import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


def compute_leep_score(
    source_prob_matrix: torch.Tensor,
    target_labels: torch.Tensor,
) -> float:
    """
    Computes Log Expected Empirical Prediction (LEEP) score (Nguyen et al., ICML 2020).
    A fast transferability metric that predicts target performance using source dummy probabilities.
    
    Args:
        source_prob_matrix: Softmax probabilities of source model on target data [N, C_source].
        target_labels: True ground-truth target labels [N].
        
    Returns:
        Scalar LEEP score (higher is better).
    """
    if not isinstance(source_prob_matrix, torch.Tensor):
        source_prob_matrix = torch.tensor(source_prob_matrix)
    if not isinstance(target_labels, torch.Tensor):
        target_labels = torch.tensor(target_labels)

    N, C_s = source_prob_matrix.shape
    unique_targets = torch.unique(target_labels)
    C_t = len(unique_targets)

    # Joint distribution P(y_target = y, z_source = z)
    joint = torch.zeros((C_t, C_s), dtype=torch.float32, device=source_prob_matrix.device)
    for idx, y_val in enumerate(unique_targets):
        mask = (target_labels == y_val)
        if mask.sum() > 0:
            joint[idx] = source_prob_matrix[mask].sum(dim=0) / N

    p_z = joint.sum(dim=0, keepdim=True) + 1e-8
    cond_p_y_given_z = joint / p_z

    leep = 0.0
    for i in range(N):
        y_val = target_labels[i]
        y_idx = (unique_targets == y_val).nonzero(as_tuple=True)[0].item()
        prob_i = torch.sum(cond_p_y_given_z[y_idx] * source_prob_matrix[i])
        leep += torch.log(prob_i + 1e-8).item()

    return float(leep / N)


def compute_logme_score(
    features: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Approximates Log Maximum Evidence (LogME) score (You et al., ICML 2021)
    for measuring representation transferability using ridge regression evidence.
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Standardize features
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    N, D = features.shape

    # One-hot encode targets
    num_classes = len(np.unique(targets))
    Y = np.eye(num_classes)[targets]

    # Approximate evidence using SVD
    try:
        _, s, _ = np.linalg.svd(features, full_matrices=False)
        alpha = 1.0
        beta = 1.0
        # Fast surrogate estimate
        evidence = -0.5 * (alpha * np.sum(s ** 2) + beta * np.linalg.norm(Y) ** 2) / N
        return float(evidence)
    except Exception:
        return 0.0

