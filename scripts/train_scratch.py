#!/usr/bin/env python3
"""
scripts/train_scratch.py
Baseline: Trains lightweight vision backbones from random initialization (scratch).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
from src.utils.config import load_config
from src.models.mobilenet_v3 import build_mobilenet_v3
from src.models.shufflenet_v2 import build_shufflenet_v2
from src.data.dataset_loader import get_dataset_loaders
from src.training.trainer import UnifiedTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train from Scratch Baseline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name override")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--subset", type=int, default=None, help="Debug subset")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="Resume training (auto or path to checkpoint)")
    parser.add_argument("--drive-backup", nargs="?", const="auto", default=None, help="Auto-sync checkpoints to Google Drive")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Overrides
    if args.dataset: cfg.data.dataset = args.dataset
    if args.epochs: cfg.training.epochs = args.epochs
    if args.batch_size: cfg.data.batch_size = args.batch_size
    if args.lr: cfg.training.lr = args.lr
    if args.subset: cfg.data.debug_subset = args.subset
    if args.device: cfg.experiment.device = args.device

    output_dir = os.path.join(cfg.experiment.output_dir, "scratch")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print(f"🛠️  BASELINE EXPERIMENT: Train from Scratch (Random Init)")
    print(f"   Dataset:  {cfg.data.dataset}")
    print(f"   Backbone: {cfg.model.backbone}")
    print(f"   Epochs:   {cfg.training.epochs}")
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

    # Build model with pretrained=False
    if "mobilenet" in cfg.model.backbone:
        model = build_mobilenet_v3(
            variant=cfg.model.backbone,
            pretrained=False,
            num_classes=num_classes,
        )
    else:
        model = build_shufflenet_v2(
            variant=cfg.model.backbone,
            pretrained=False,
            num_classes=num_classes,
        )

    # Optimizer
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
        scheduler=scheduler,
        device=cfg.experiment.device,
        output_dir=output_dir,
        is_transferguard=False,
        resume=args.resume,
        drive_backup=args.drive_backup,
    )

    results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.training.epochs,
    )
    print(f"\n✅ Scratch baseline complete. Best Val Accuracy: {results['best_val_acc1']:.2f}%")


if __name__ == "__main__":
    main()
