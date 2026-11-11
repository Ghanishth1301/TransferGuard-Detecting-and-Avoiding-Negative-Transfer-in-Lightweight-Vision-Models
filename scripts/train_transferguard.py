#!/usr/bin/env python3
"""
scripts/train_transferguard.py
Proposed Method: TransferGuard Adaptive Routing & Dynamic Layer Freezing.
Prevents negative transfer by locking or bypassing mismatched deep representations.

Authors: Arvind Yadav, Ghanishth Saini, Aakash Jagtap
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
from src.utils.config import load_config
from src.models.mobilenet_v3 import build_mobilenet_v3
from src.models.shufflenet_v2 import build_shufflenet_v2
from src.models.adaptive_routing import AdaptiveRoutingModel
from src.data.dataset_loader import get_dataset_loaders
from src.training.loss import TransferGuardLoss
from src.training.trainer import UnifiedTrainer
from src.utils.visualization import plot_layer_freezing_profile


def parse_args():
    parser = argparse.ArgumentParser(description="Train Proposed TransferGuard Adaptive Routing")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name override (svhn, cifar100, flowers102, synthetic)")
    parser.add_argument("--threshold", type=float, default=None, help="Freeze threshold (tau_freeze)")
    parser.add_argument("--bypass-threshold", type=float, default=None, help="Bypass threshold (tau_bypass)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="Resume training (auto or path to checkpoint)")
    parser.add_argument("--drive-backup", nargs="?", const="auto", default=None, help="Auto-sync checkpoints to Google Drive")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Overrides
    if args.dataset: cfg.data.dataset = args.dataset
    if args.threshold: cfg.transferguard.freeze_threshold = args.threshold
    if args.bypass_threshold: cfg.transferguard.bypass_threshold = args.bypass_threshold
    if args.epochs: cfg.training.epochs = args.epochs
    if args.batch_size: cfg.data.batch_size = args.batch_size
    if args.lr: cfg.training.lr = args.lr
    if args.subset: cfg.data.debug_subset = args.subset
    if args.device: cfg.experiment.device = args.device

    output_dir = os.path.join(cfg.experiment.output_dir, "transferguard_adaptive")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print(f"🛡️  PROPOSED METHOD: TransferGuard Adaptive Layer Freezing & Routing")
    print(f"   Dataset:          {cfg.data.dataset}")
    print(f"   Backbone:         {cfg.model.backbone} (ImageNet Pre-trained)")
    print(f"   Freeze Threshold: {cfg.transferguard.freeze_threshold}")
    print(f"   Bypass Threshold: {cfg.transferguard.bypass_threshold}")
    print(f"   Epochs:           {cfg.training.epochs}")
    print("=" * 65)

    # Load data
    train_loader, val_loader, num_classes = get_dataset_loaders(
        dataset_name=cfg.data.dataset,
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
        debug_subset=cfg.data.debug_subset,
    )

    # Build backbone
    if "mobilenet" in cfg.model.backbone:
        backbone = build_mobilenet_v3(
            variant=cfg.model.backbone,
            pretrained=True,
            num_classes=num_classes,
        )
    else:
        backbone = build_shufflenet_v2(
            variant=cfg.model.backbone,
            pretrained=True,
            num_classes=num_classes,
        )

    # Wrap in Adaptive Routing Model
    model = AdaptiveRoutingModel(
        backbone=backbone,
        freeze_threshold=cfg.transferguard.freeze_threshold,
        bypass_threshold=cfg.transferguard.bypass_threshold,
    )

    criterion = TransferGuardLoss(
        label_smoothing=0.05,
        regularization_lambda=cfg.transferguard.regularization_lambda,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6
    )

    trainer = UnifiedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=cfg.experiment.device,
        output_dir=output_dir,
        is_transferguard=True,
        resume=args.resume,
        drive_backup=args.drive_backup,
    )

    results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.training.epochs,
    )

    # Output final routing decisions
    print("\n🏁 Final TransferGuard Routing Summary:")
    summary = model.get_routing_summary()
    for stage, details in summary.items():
        print(f"   Stage [{stage:12s}]: Score = {details['score']:.4f} -> Action: {details['action'].upper()}")

    # Save structured routing profile JSON for downstream plotting and evaluation
    routing_data = {
        "dataset": cfg.data.dataset,
        "backbone": cfg.model.backbone,
        "freeze_threshold": cfg.transferguard.freeze_threshold,
        "bypass_threshold": cfg.transferguard.bypass_threshold,
        "running_scores": {k: float(v) for k, v in model.running_scores.items()},
        "routing_actions": {k: v["action"] for k, v in summary.items()},
        "history_scores": model.history_scores,
    }
    routing_json_path = os.path.join(output_dir, "routing_profile.json")
    with open(routing_json_path, "w") as f:
        json.dump(routing_data, f, indent=2)
    print(f"📄 Saved routing decisions data to: {routing_json_path}")

    # Save summary eval JSON
    eval_summary = {
        "method": "transferguard",
        "dataset": cfg.data.dataset,
        "backbone": cfg.model.backbone,
        "epochs": cfg.training.epochs,
        "best_val_acc1": results.get("best_val_acc1", 0.0),
        "total_time": results.get("total_time", 0.0),
    }
    with open(os.path.join(output_dir, "eval_summary.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)

    # Save layer routing profile figure
    profile_path = os.path.join(output_dir, "transferguard_layer_profile.png")
    plot_layer_freezing_profile(
        layer_scores=model.running_scores,
        freeze_threshold=cfg.transferguard.freeze_threshold,
        bypass_threshold=cfg.transferguard.bypass_threshold,
        save_path=profile_path,
        title=f"TransferGuard Routing Profile: {cfg.data.dataset.upper()} ({cfg.model.backbone})",
    )
    print(f"📊 Saved layer routing visualization to: {profile_path}")
    print(f"\n✅ TransferGuard training complete. Best Val Accuracy: {results['best_val_acc1']:.2f}%")


if __name__ == "__main__":
    main()
