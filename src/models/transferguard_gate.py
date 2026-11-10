import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class TransferGuardGate(nn.Module):
    """
    Lightweight Gating Unit for a single architectural stage.
    Maps intermediate stage representation z_l in R^C to a scalar transferability
    score S_l in [0, 1].
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
            x: Feature representation of shape [B, C].
        Returns:
            Scalar score of shape [B, 1] or mean score for the batch.
        """
        return self.mlp(x)


class LayerGatingController(nn.Module):
    """
    Controller that manages gating across all backbone stages.
    Evaluates layer-wise transferability scores and yields dynamic routing decisions.
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

        self.gates = nn.ModuleDict({
            name: TransferGuardGate(dim, reduction=reduction)
            for name, dim in stage_channels.items()
        })

    def forward(self, stage_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Computes transferability scores for all stages.
        Returns dict mapping stage_name -> tensor of shape [B, 1].
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
        Classifies each stage action based on transferability score:
        - "freeze": Score >= freeze_threshold (High transferability -> lock pre-trained weights)
        - "tune": bypass_threshold <= Score < freeze_threshold (Moderate -> plastic fine-tuning)
        - "bypass": Score < bypass_threshold (Severe mismatch -> bypass/reset conflicting features)
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
        Encourages decisive gating (bimodal distribution near 0 or 1, avoiding vague indecision)
        using binary entropy regularization.
        """
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        eps = 1e-6
        for score in scores.values():
            p = torch.clamp(score, eps, 1.0 - eps)
            # Binary entropy: - (p*log(p) + (1-p)*log(1-p)) -> minimizing encourages 0 or 1
            entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean()
            loss = loss + entropy
        return loss / max(1, len(scores))

