"""
src/models/mobilenet_v3.py
Modular PyTorch wrapper for MobileNetV3 (Small & Large) vision backbones.
Provides forward activation hooks for intermediate stage feature extraction,
spatial average pooling, and selective stage-wise parameter freezing.

Authors: Arvind Yadav, Ghanishth Saini, Aakash Jagtap
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights


class MobileNetV3Wrapper(nn.Module):
    """
    Wrapper for MobileNetV3 (Small or Large) with intermediate stage hooks,
    representation extraction, and selective layer freezing capabilities.

    Architecture stages are grouped into 5 logical blocks:
      - Stem: Initial 3x3 conv layer (edges & basic gabor features)
      - Stage Early: Inverted residual blocks 1-2 (corners, simple textures)
      - Stage Mid-1: Inverted residual blocks 3-5 (complex textures, motifs)
      - Stage Mid-2: Inverted residual blocks 6-8 (parts & semantic patterns)
      - Stage Deep: Final inverted residual blocks (class-specific representations)
    """

    def __init__(
        self,
        variant: str = "mobilenet_v3_small",
        pretrained: bool = True,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.variant = variant.lower()
        self.pretrained = pretrained
        self.num_classes = num_classes

        # Load pre-trained or randomly initialized backbone from torchvision
        if self.variant == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[0].in_features
        elif self.variant == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_large(weights=weights)
            in_features = self.backbone.classifier[0].in_features
        else:
            raise ValueError(f"Unsupported variant: {variant}. Choose 'mobilenet_v3_small' or 'mobilenet_v3_large'.")

        # Re-initialize the final classification projection head for target domain num_classes
        last_channel = self.backbone.classifier[3].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        # Map inverted residual blocks into 5 progressive representation stages
        num_blocks = len(self.backbone.features)
        step = max(1, num_blocks // 5)
        self.stage_indices = {
            "stem": 0,
            "stage_early": min(step, num_blocks - 1),
            "stage_mid_1": min(step * 2, num_blocks - 1),
            "stage_mid_2": min(step * 3, num_blocks - 1),
            "stage_deep": num_blocks - 1,
        }

        self._hook_features: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """
        Registers forward hooks on stage boundary modules.
        During forward inference, spatial feature maps [B, C, H, W] are globally average-pooled
        into compact [B, C] vector representations for CKA analysis and gating.
        """
        def get_hook(name: str):
            def hook(module, input, output):
                if output.dim() == 4:
                    pooled = output.mean(dim=[2, 3])
                else:
                    pooled = output
                self._hook_features[name] = pooled
            return hook

        for name, idx in self.stage_indices.items():
            hook_handle = self.backbone.features[idx].register_forward_hook(get_hook(name))
            self._hooks.append(hook_handle)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning target class logits [B, num_classes]."""
        return self.backbone(x)

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass returning both final class logits and pooled intermediate stage representations.
        Returns:
            logits: Tensor of shape [B, num_classes]
            features: Dict mapping stage_name -> Tensor [B, C_l]
        """
        self._hook_features.clear()
        logits = self.backbone(x)
        features = {k: v for k, v in self._hook_features.items()}
        return logits, features

    def freeze_stages(self, stages_to_freeze: List[str]):
        """
        Freezes backbone layers up to the highest specified stage index.
        Freezing sets requires_grad = False, eliminating backprop gradient computation.
        """
        stage_order = list(self.stage_indices.keys())
        frozen_indices = set()
        for stage in stages_to_freeze:
            if stage in self.stage_indices:
                idx = self.stage_indices[stage]
                stage_pos = stage_order.index(stage)
                for prev_s in stage_order[: stage_pos + 1]:
                    frozen_indices.add(self.stage_indices[prev_s])

        max_frozen_idx = max(frozen_indices) if frozen_indices else -1
        for i, block in enumerate(self.backbone.features):
            requires_grad = i > max_frozen_idx
            for param in block.parameters():
                param.requires_grad = requires_grad

    def freeze_backbone(self):
        """Freezes all feature extraction blocks (for linear probing evaluation)."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreezes all parameters across backbone and classifier for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_stage_channels(self, sample_input: Optional[torch.Tensor] = None) -> Dict[str, int]:
        """Calculates channel dimension C_l for each intermediate stage representation."""
        if sample_input is None:
            sample_input = torch.zeros(1, 3, 224, 224, device=next(self.parameters()).device)
        with torch.no_grad():
            _, feats = self.forward_with_features(sample_input)
        return {k: v.shape[1] for k, v in feats.items()}


def build_mobilenet_v3(
    variant: str = "mobilenet_v3_small",
    pretrained: bool = True,
    num_classes: int = 10,
    dropout: float = 0.2,
) -> MobileNetV3Wrapper:
    return MobileNetV3Wrapper(
        variant=variant,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout,
    )

