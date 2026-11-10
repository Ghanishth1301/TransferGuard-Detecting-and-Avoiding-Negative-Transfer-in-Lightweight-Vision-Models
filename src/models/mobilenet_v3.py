import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights


class MobileNetV3Wrapper(nn.Module):
    """
    Wrapper for MobileNetV3 (Small or Large) with intermediate stage hooks,
    representation extraction, and selective layer freezing capabilities.
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

        # Load backbone
        if self.variant == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[0].in_features
        elif self.variant == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_large(weights=weights)
            in_features = self.backbone.classifier[0].in_features
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        # Replace classification head
        last_channel = self.backbone.classifier[3].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        # Stage definitions for hook extractions
        # MobileNetV3 features is a Sequential of inverted residual blocks (0 to len-1)
        # Small has 13 feature blocks (0..12), Large has 17 feature blocks (0..16)
        num_blocks = len(self.backbone.features)
        # Group into 5 logical stages: Stem, Early, Mid-Early, Mid-Late, Deep
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
        """Registers forward hooks on stage boundary modules."""
        def get_hook(name: str):
            def hook(module, input, output):
                # Spatially pool feature map to [B, C]
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
        """Standard forward pass returning logits."""
        return self.backbone(x)

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass returning both final class logits and intermediate stage representations.
        """
        self._hook_features.clear()
        logits = self.backbone(x)
        features = {k: v for k, v in self._hook_features.items()}
        return logits, features

    def freeze_stages(self, stages_to_freeze: List[str]):
        """Freezes backbone blocks up to the specified stages."""
        stage_order = list(self.stage_indices.keys())
        frozen_indices = set()
        for stage in stages_to_freeze:
            if stage in self.stage_indices:
                idx = self.stage_indices[stage]
                stage_pos = stage_order.index(stage)
                # Freeze from 0 up to idx
                for prev_s in stage_order[: stage_pos + 1]:
                    frozen_indices.add(self.stage_indices[prev_s])

        max_frozen_idx = max(frozen_indices) if frozen_indices else -1
        for i, block in enumerate(self.backbone.features):
            requires_grad = i > max_frozen_idx
            for param in block.parameters():
                param.requires_grad = requires_grad

    def freeze_backbone(self):
        """Freezes all features (linear probe mode)."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreezes all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_stage_channels(self, sample_input: Optional[torch.Tensor] = None) -> Dict[str, int]:
        """Calculates channel dimension for each stage representation."""
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

