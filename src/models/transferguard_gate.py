"""
src/models/transferguard_gate.py
TransferGuard Micro-Gating Network & Dynamic Layer Routing Controller.
Maps intermediate stage representations to scalar transferability scores S_l in [0, 1]
and computes adaptive routing decisions (Freeze, Bypass, or Tune) to avoid negative transfer.

Authors: Arvind Yadav, Ghanishth Saini, Aakash Jagtap
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class TransferGuardGate(nn.Module):
    """
    Lightweight Gating Unit attached to an architectural stage.
    Maps an intermediate pooled stage representation z_l in R^{C_l} to a scalar
    transferability score S_l in [0, 1].

    Architecture:
      Linear(C_l, C_l // reduction) -> LayerNorm -> ReLU -> Linear(C_l // reduction, 1) -> Sigmoid()
    """

    def __init__(self, in_features: int, reduction: int = 8):
        super().__init__()
        hidden_dim = max(8, in_features // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatially pooled feature representation of shape [B, C_l].
        Returns:
            Scalar transferability score tensor of shape [B, 1].
        """
        return self.mlp(x)


class LayerGatingController(nn.Module):
    """
    Multi-stage Gating Controller that coordinates micro-gates across all backbone stages.
    Evaluates layer-wise transferability scores, computes the entropy regularization loss,
    and returns categorical routing decisions.
    """

    def __init__(
        self,
        stage_channels: Dict[str, int],
        reduction: int = 8,
        freeze_threshold: float = 0.65,
        bypass_threshold: float = 0.35,
    ):
        super().__init__()
        self.stage_names = list(stage_channels.keys())
        self.freeze_threshold = freeze_threshold
        self.bypass_threshold = bypass_threshold

        # Dedicated micro-gate module for each stage
        self.gates = nn.ModuleDict({
            name: TransferGuardGate(dim, reduction=reduction)
            for name, dim in stage_channels.items()
        })

    def forward(self, stage_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Computes transferability scores for each stage from feature maps.
        Returns dict mapping stage_name -> tensor [B, 1].
        """
        scores = {}
        for name, feats in stage_features.items():
            if name in self.gates:
                scores[name] = self.gates[name](feats)
        return scores

    def get_mean_scores(self, stage_features: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Returns detached scalar mean scores per stage for the current batch."""
        scores = self.forward(stage_features)
        return {k: float(v.mean().item()) for k, v in scores.items()}

    def get_routing_actions(self, mean_scores: Dict[str, float]) -> Dict[str, str]:
        """
        Maps continuous transferability scores S_l in [0, 1] to discrete routing decisions:
        - "freeze": Score >= freeze_threshold (High transferability: lock pre-trained weights to preserve features)
        - "tune": bypass_threshold < Score < freeze_threshold (Moderate: standard fine-tuning)
        - "bypass": Score <= bypass_threshold (Severe mismatch: bypass/adapt conflicting deep representations)
        """
        actions = {}
        for name, score in mean_scores.items():
            if score >= self.freeze_threshold:
                actions[name] = "freeze"
            elif score <= self.bypass_threshold:
                actions[name] = "bypass"
            else:
                actions[name] = "tune"
        return actions

    def compute_gate_regularization_loss(self, scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Bimodal binary entropy regularization:
          H(p) = - [p * log(p) + (1 - p) * log(1 - p)]
        Minimizing binary entropy penalizes indecisive gate scores near 0.5,
        pushing scores decisively toward 1.0 (freeze) or 0.0 (bypass).
        All computations are cast to float32 to prevent underflow in FP16 AMP.
        """
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        eps = 1e-5
        for score in scores.values():
            p = score.float()
            p = torch.clamp(p, eps, 1.0 - eps)
            entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean()
            loss = loss + entropy.to(score.dtype)
        return loss / max(1, len(scores))

