import pytest
import os
import shutil
import torch
import torch.optim as optim
from src.data.dataset_loader import get_dataset_loaders
from src.models.mobilenet_v3 import build_mobilenet_v3
from src.models.adaptive_routing import AdaptiveRoutingModel
from src.training.trainer import UnifiedTrainer
from src.training.loss import TransferGuardLoss


def test_end_to_end_cpu_dryrun(tmp_path):
    """
    Executes a fast 1-epoch end-to-end dry run on CPU using synthetic mini-batches.
    Verifies data loader -> forward pass -> loss -> backward -> dynamic routing update -> checkpoint.
    """
    test_out = str(tmp_path / "dryrun_out")

    # 1. Synthetic dataset
    train_loader, val_loader, num_classes = get_dataset_loaders(
        dataset_name="synthetic",
        batch_size=8,
        debug_subset=16,
        image_size=64, # small size for ultra-fast CPU test
    )

    # 2. Build model
    backbone = build_mobilenet_v3(
        variant="mobilenet_v3_small",
        pretrained=False,
        num_classes=num_classes,
    )
    model = AdaptiveRoutingModel(backbone=backbone, freeze_threshold=0.6)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = TransferGuardLoss()

    trainer = UnifiedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
        output_dir=test_out,
        is_transferguard=True,
    )

    results = trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=1)

    assert "best_val_acc1" in results
    assert os.path.exists(os.path.join(test_out, "train_history.json"))
    assert os.path.exists(os.path.join(test_out, "checkpoints", "best_model.pth"))

