"""
src/training/loss.py
Composite loss objective for TransferGuard:
Total Loss = Classification Cross-Entropy + lambda * Gate Entropy Regularization.

Authors: Arvind Yadav, Ghanishth Saini, Aakash Jagtap
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class TransferGuardLoss(nn.Module):
    """
    Combined objective for training with TransferGuard:
    L = L_classification(logits, targets) + lambda * L_gate_regularization(gate_scores)
    """

    def __init__(
        self,
        label_smoothing: float = 0.05,
        regularization_lambda: float = 0.01,
    ):
        super().__init__()
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.reg_lambda = regularization_lambda

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gate_scores: Optional[Dict[str, torch.Tensor]] = None,
        controller: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict containing:
            - total_loss: Tensor
            - cls_loss: Tensor
            - reg_loss: Tensor
        """
        cls_loss = self.classification_loss(logits, targets)
        reg_loss = torch.tensor(0.0, device=logits.device)

        if gate_scores is not None and controller is not None and self.reg_lambda > 0:
            reg_loss = controller.compute_gate_regularization_loss(gate_scores)

        total_loss = cls_loss + self.reg_lambda * reg_loss

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "reg_loss": reg_loss.detach(),
        }

