"""
src/models/adaptive_routing.py
Adaptive Routing Controller for TransferGuard.
Pairs lightweight vision backbones (MobileNetV3, ShuffleNetV2) with the
LayerGatingController to dynamically steer feature representations during training.

Authors: Arvind Yadav, Ghanishth Saini, Aakash Jagtap
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from .mobilenet_v3 import MobileNetV3Wrapper
from .shufflenet_v2 import ShuffleNetV2Wrapper
from .transferguard_gate import LayerGatingController


class AdaptiveRoutingModel(nn.Module):
    """
    Integrates a lightweight vision backbone with the TransferGuard
    layer gating and adaptive routing controller.

    Workflow:
      1. Forward pass extracts class predictions and intermediate pooled features across 5 stages.
      2. LayerGatingController computes stage-wise transferability scores S_l in [0, 1].
      3. An Exponential Moving Average (EMA) tracks running stage transferability across batches.
      4. update_dynamic_routing() applies threshold actions (Freeze, Bypass, or Tune) to the backbone.
    """

    def __init__(
        self,
        backbone: Union[MobileNetV3Wrapper, ShuffleNetV2Wrapper],
        freeze_threshold: float = 0.65,
        bypass_threshold: float = 0.35,
        gate_reduction: int = 8,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_threshold = freeze_threshold
        self.bypass_threshold = bypass_threshold

        # Probe stage channel dimensions automatically with a sample tensor
        sample_input = torch.zeros(2, 3, 224, 224)
        stage_channels = self.backbone.get_stage_channels(sample_input)

        self.controller = LayerGatingController(
            stage_channels=stage_channels,
            reduction=gate_reduction,
            freeze_threshold=freeze_threshold,
            bypass_threshold=bypass_threshold,
        )

        # Initialize running transferability scores at neutral 0.5
        self.running_scores: Dict[str, float] = {k: 0.5 for k in stage_channels.keys()}
        self.history_scores: List[Dict[str, float]] = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass returning both classification logits and gate score tensors.
        Returns:
            logits: Shape [B, num_classes]
            gate_scores: Dict mapping stage_name -> Tensor [B, 1]
        """
        logits, features = self.backbone.forward_with_features(x)
        gate_scores = self.controller(features)
        return logits, gate_scores

    def update_dynamic_routing(self, alpha: float = 0.1) -> Dict[str, str]:
        """
        Evaluates running transferability scores and applies freezing actions to the backbone.
        Stages classified as 'freeze' have their requires_grad set to False,
        locking in pre-trained features and accelerating convergence.
        """
        with torch.no_grad():
            stage_names = list(self.running_scores.keys())
            actions = self.controller.get_routing_actions(self.running_scores)

            stages_to_freeze = [k for k, act in actions.items() if act == "freeze"]
            self.backbone.freeze_stages(stages_to_freeze)

            self.history_scores.append(dict(self.running_scores))
            return actions

    def record_batch_scores(self, gate_scores: Dict[str, torch.Tensor], momentum: float = 0.9):
        """
        Updates running EMA of gate transferability scores using current mini-batch:
          S_running = momentum * S_running + (1 - momentum) * S_batch
        """
        with torch.no_grad():
            for name, score_tensor in gate_scores.items():
                batch_mean = float(score_tensor.mean().item())
                if name in self.running_scores:
                    self.running_scores[name] = (
                        momentum * self.running_scores[name] + (1.0 - momentum) * batch_mean
                    )
                else:
                    self.running_scores[name] = batch_mean

    def get_routing_summary(self) -> Dict[str, Dict[str, Union[float, str]]]:
        """Returns structured dictionary of stage scores and current routing actions."""
        actions = self.controller.get_routing_actions(self.running_scores)
        summary = {}
        for k in self.running_scores:
            summary[k] = {
                "score": round(self.running_scores[k], 4),
                "action": actions.get(k, "tune"),
            }
        return summary

