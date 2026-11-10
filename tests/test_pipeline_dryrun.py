import pytest
import os
import json
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
        image_size=64,
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
    assert os.path.exists(os.path.join(test_out, "checkpoints", "latest_model.pth"))


def test_resumable_training(tmp_path):
    """
    Verifies that training can be cleanly resumed from a saved checkpoint:
    1. Trains epoch 1 and saves latest_model.pth.
    2. Initializes new trainer with resume=True.
    3. Verifies trainer starts at epoch 2 and completes training with both epochs in history.
    """
    test_out = str(tmp_path / "resume_out")

    train_loader, val_loader, num_classes = get_dataset_loaders(
        dataset_name="synthetic",
        batch_size=8,
        debug_subset=16,
        image_size=64,
    )

    # First session: Train 1 epoch
    backbone_1 = build_mobilenet_v3(variant="mobilenet_v3_small", pretrained=False, num_classes=num_classes)
    model_1 = AdaptiveRoutingModel(backbone=backbone_1, freeze_threshold=0.6)
    opt_1 = optim.AdamW(model_1.parameters(), lr=1e-3)

    trainer_1 = UnifiedTrainer(
        model=model_1,
        optimizer=opt_1,
        device="cpu",
        output_dir=test_out,
        is_transferguard=True,
    )
    trainer_1.fit(train_loader=train_loader, val_loader=val_loader, epochs=1)

    assert os.path.exists(os.path.join(test_out, "checkpoints", "latest_model.pth"))

    # Second session: Resume from checkpoint and train up to epoch 2
    backbone_2 = build_mobilenet_v3(variant="mobilenet_v3_small", pretrained=False, num_classes=num_classes)
    model_2 = AdaptiveRoutingModel(backbone=backbone_2, freeze_threshold=0.6)
    opt_2 = optim.AdamW(model_2.parameters(), lr=1e-3)

    trainer_2 = UnifiedTrainer(
        model=model_2,
        optimizer=opt_2,
        device="cpu",
        output_dir=test_out,
        is_transferguard=True,
        resume="auto",
    )

    assert trainer_2.start_epoch == 1

    trainer_2.fit(train_loader=train_loader, val_loader=val_loader, epochs=2)

    # Verify history contains both epoch 1 and epoch 2
    with open(os.path.join(test_out, "train_history.json"), "r") as f:
        history = json.load(f)

    assert len(history) == 2
    assert history[0]["epoch"] == 1
    assert history[1]["epoch"] == 2

    # Verify already-completed check
    results_already = trainer_2.fit(train_loader=train_loader, val_loader=val_loader, epochs=2)
    assert results_already["total_time"] == 0.0


def test_drive_backup_sync_and_restore(tmp_path):
    """Verifies that Google Drive backup persists checkpoints and restores after container wipe."""
    mock_drive = str(tmp_path / "google_drive_mock" / "TransferGuard_Backups")
    local_out = str(tmp_path / "results" / "svhn_mobilenetv3" / "scratch")

    train_loader, val_loader, num_classes = get_dataset_loaders(
        dataset_name="synthetic",
        batch_size=8,
        debug_subset=16,
        image_size=64,
    )

    # Session 1: Train 1 epoch with drive_backup enabled
    model_1 = build_mobilenet_v3(variant="mobilenet_v3_small", pretrained=False, num_classes=num_classes)
    opt_1 = optim.AdamW(model_1.parameters(), lr=1e-3)

    trainer_1 = UnifiedTrainer(
        model=model_1,
        optimizer=opt_1,
        device="cpu",
        output_dir=local_out,
        is_transferguard=False,
        drive_backup=mock_drive,
    )
    trainer_1.fit(train_loader=train_loader, val_loader=val_loader, epochs=1)

    # Verify files synced to mock Google Drive
    drive_exp_dir = os.path.join(mock_drive, "svhn_mobilenetv3", "scratch")
    assert os.path.exists(os.path.join(drive_exp_dir, "train_history.json"))
    assert os.path.exists(os.path.join(drive_exp_dir, "checkpoints", "latest_model.pth"))

    # Simulate catastrophic Colab disconnection/container teardown: wipe local results completely
    import shutil
    shutil.rmtree(local_out)
    assert not os.path.exists(local_out)

    # Session 2: Fresh container starts with empty local directory, connects Drive and resumes
    model_2 = build_mobilenet_v3(variant="mobilenet_v3_small", pretrained=False, num_classes=num_classes)
    opt_2 = optim.AdamW(model_2.parameters(), lr=1e-3)

    trainer_2 = UnifiedTrainer(
        model=model_2,
        optimizer=opt_2,
        device="cpu",
        output_dir=local_out,
        is_transferguard=False,
        resume="auto",
        drive_backup=mock_drive,
    )

    # Verify that checkpoints were restored from Drive and epoch resumed from 1
    assert os.path.exists(os.path.join(local_out, "checkpoints", "latest_model.pth"))
    assert trainer_2.start_epoch == 1

    # Train epoch 2
    trainer_2.fit(train_loader=train_loader, val_loader=val_loader, epochs=2)

    # Verify that mock drive now has the updated epoch 2 history
    with open(os.path.join(drive_exp_dir, "train_history.json"), "r") as f:
        history = json.load(f)
    assert len(history) == 2
    assert history[0]["epoch"] == 1
    assert history[1]["epoch"] == 2

