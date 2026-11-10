import pytest
import torch
from src.models.mobilenet_v3 import build_mobilenet_v3
from src.models.shufflenet_v2 import build_shufflenet_v2
from src.models.transferguard_gate import LayerGatingController
from src.models.adaptive_routing import AdaptiveRoutingModel


def test_mobilenet_v3_feature_hooks():
    """Tests feature extraction across stages for MobileNetV3-Small."""
    model = build_mobilenet_v3(variant="mobilenet_v3_small", pretrained=False, num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    logits, features = model.forward_with_features(x)

    assert logits.shape == (2, 10)
    assert "stem" in features
    assert "stage_early" in features
    assert "stage_deep" in features
    assert features["stem"].shape[0] == 2


def test_shufflenet_v2_feature_hooks():
    """Tests feature extraction across stages for ShuffleNetV2."""
    model = build_shufflenet_v2(variant="shufflenet_v2_x1_0", pretrained=False, num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    logits, features = model.forward_with_features(x)

    assert logits.shape == (2, 10)
    assert "stem" in features
    assert "stage_deep" in features


def test_transferguard_controller_gating():
    """Tests gating score generation and range bounds [0, 1]."""
    stage_channels = {"stem": 16, "stage_early": 24, "stage_deep": 576}
    controller = LayerGatingController(stage_channels=stage_channels, freeze_threshold=0.7)

    features = {
        name: torch.randn(4, dim) for name, dim in stage_channels.items()
    }

    scores = controller(features)
    for name, s in scores.items():
        assert s.shape == (4, 1)
        assert torch.all(s >= 0.0) and torch.all(s <= 1.0)

    actions = controller.get_routing_actions({"stem": 0.85, "stage_deep": 0.25})
    assert actions["stem"] == "freeze"
    assert actions["stage_deep"] == "bypass"


def test_adaptive_routing_freezing():
    """Tests that freezing action locks corresponding backbone parameters."""
    backbone = build_mobilenet_v3(variant="mobilenet_v3_small", pretrained=False, num_classes=10)
    model = AdaptiveRoutingModel(backbone=backbone, freeze_threshold=0.6)

    # Force stem score high to trigger freeze
    model.running_scores["stem"] = 0.95
    model.running_scores["stage_deep"] = 0.10

    actions = model.update_dynamic_routing()
    assert actions["stem"] == "freeze"

    # Verify stem layer parameters have requires_grad=False
    stem_params = list(backbone.backbone.features[0].parameters())
    assert all(not p.requires_grad for p in stem_params)

