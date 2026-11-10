import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import torchvision.models as models
from torchvision.models import ShuffleNet_V2_X1_0_Weights


class ShuffleNetV2Wrapper(nn.Module):
    """
    Wrapper for ShuffleNetV2 1.0x with stage hooks, representation extraction,
    and selective layer freezing.
    """

    def __init__(
        self,
        variant: str = "shufflenet_v2_x1_0",
        pretrained: bool = True,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.variant = variant.lower()
        self.pretrained = pretrained
        self.num_classes = num_classes

        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        self.backbone = models.shufflenet_v2_x1_0(weights=weights)

        # Replace classifier
        in_features = self.backbone.fc.in_features
        if dropout > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

        self._hook_features: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward hooks on ShuffleNet stages."""
        def get_hook(name: str):
            def hook(module, input, output):
                if output.dim() == 4:
                    pooled = output.mean(dim=[2, 3])
                else:
                    pooled = output
                self._hook_features[name] = pooled
            return hook

        stages = {
            "stem": self.backbone.conv1,
            "stage_2": self.backbone.stage2,
            "stage_3": self.backbone.stage3,
            "stage_4": self.backbone.stage4,
            "stage_deep": self.backbone.conv5,
        }

        for name, module in stages.items():
            handle = module.register_forward_hook(get_hook(name))
            self._hooks.append(handle)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._hook_features.clear()
        logits = self.backbone(x)
        features = {k: v for k, v in self._hook_features.items()}
        return logits, features

    def freeze_stages(self, stages_to_freeze: List[str]):
        """Selectively freezes stages."""
        stage_map = {
            "stem": [self.backbone.conv1, self.backbone.maxpool],
            "stage_2": [self.backbone.stage2],
            "stage_3": [self.backbone.stage3],
            "stage_4": [self.backbone.stage4],
            "stage_deep": [self.backbone.conv5],
        }
        for stage_name, modules in stage_map.items():
            freeze = stage_name in stages_to_freeze
            for mod in modules:
                for param in mod.parameters():
                    param.requires_grad = not freeze

    def freeze_backbone(self):
        """Freezes all feature extraction layers (linear probe mode)."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def get_stage_channels(self, sample_input: Optional[torch.Tensor] = None) -> Dict[str, int]:
        if sample_input is None:
            sample_input = torch.zeros(1, 3, 224, 224, device=next(self.parameters()).device)
        with torch.no_grad():
            _, feats = self.forward_with_features(sample_input)
        return {k: v.shape[1] for k, v in feats.items()}


def build_shufflenet_v2(
    variant: str = "shufflenet_v2_x1_0",
    pretrained: bool = True,
    num_classes: int = 10,
    dropout: float = 0.2,
) -> ShuffleNetV2Wrapper:
    return ShuffleNetV2Wrapper(
        variant=variant,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout,
    )

