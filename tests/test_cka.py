import pytest
import torch
import numpy as np
from src.metrics.cka import linear_cka, kernel_cka, compute_layer_cka_matrix


def test_linear_cka_identity():
    """CKA of any activation matrix with itself must equal 1.0."""
    torch.manual_seed(42)
    X = torch.randn(64, 128)
    score = linear_cka(X, X, unbiased=True)
    assert pytest.approx(score, abs=1e-3) == 1.0


def test_linear_cka_scaling_invariance():
    """CKA must be invariant to uniform scalar multiplication."""
    torch.manual_seed(42)
    X = torch.randn(64, 128)
    Y = 7.5 * X
    score = linear_cka(X, Y, unbiased=True)
    assert pytest.approx(score, abs=1e-3) == 1.0


def test_linear_cka_orthogonal_invariance():
    """CKA must be invariant to orthogonal transformations."""
    torch.manual_seed(42)
    X = torch.randn(64, 64)
    # Generate random orthogonal matrix via QR decomposition
    q, _ = torch.linalg.qr(torch.randn(64, 64))
    Y = torch.matmul(X, q)

    score = linear_cka(X, Y, unbiased=True)
    assert pytest.approx(score, abs=1e-2) == 1.0


def test_linear_cka_uncorrelated():
    """Independent random representations should yield low CKA score."""
    torch.manual_seed(42)
    X = torch.randn(256, 128)
    Y = torch.randn(256, 128)
    score = linear_cka(X, Y, unbiased=True)
    assert score < 0.20


def test_compute_layer_cka_matrix():
    """Tests matrix computation across multiple feature hooks."""
    torch.manual_seed(42)
    feats_a = {
        "layer_1": torch.randn(32, 64),
        "layer_2": torch.randn(32, 128),
    }
    feats_b = {
        "layer_1": feats_a["layer_1"],  # identical
        "layer_2": torch.randn(32, 128),  # random
    }

    matrix, names_a, names_b = compute_layer_cka_matrix(feats_a, feats_b)
    assert matrix.shape == (2, 2)
    assert pytest.approx(matrix[0, 0], abs=1e-3) == 1.0
    assert matrix[1, 1] < 0.30

