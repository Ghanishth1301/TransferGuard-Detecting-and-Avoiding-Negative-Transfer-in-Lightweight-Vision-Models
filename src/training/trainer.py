"""
src/training/trainer.py
Unified training engine supporting mixed-precision AMP, learning rate scheduling,
dynamic stage freezing callbacks, fault-tolerant checkpoint resumption, and metric logging.

Authors: Arvind Yadav, Ghanishth Saini, Aakash Jagtap
"""

import os
import time
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union
from torch.utils.data import DataLoader
from tqdm import tqdm
from .loss import TransferGuardLoss
from .callbacks import MetricLogger, ModelCheckpoint, DriveBackupManager
from ..models.adaptive_routing import AdaptiveRoutingModel


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[float]:
    """Computes precision@k for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


class UnifiedTrainer:
    """
    Unified Trainer handling scratch training, standard fine-tuning,
    linear probing, and TransferGuard adaptive routing with resumable training.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[TransferGuardLoss] = None,
        scheduler: Optional[Any] = None,
        device: str = "auto",
        output_dir: str = "results",
        is_transferguard: bool = False,
        resume: Union[bool, str] = False,
        drive_backup: Optional[str] = None,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion or TransferGuardLoss()
        self.scheduler = scheduler
        self.is_transferguard = is_transferguard
        self.output_dir = output_dir

        # High-performance GPU settings (Tensor Core AMP & cuDNN benchmarking)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.use_amp = (self.device.type == "cuda")
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(device_type, enabled=self.use_amp)

        os.makedirs(output_dir, exist_ok=True)
        self.logger = MetricLogger()
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=os.path.join(output_dir, "checkpoints")
        )
        self.drive_manager = DriveBackupManager(drive_backup, output_dir=output_dir)

        self.start_epoch = 0
        self.best_acc1 = 0.0

        if resume:
            self._handle_resume(resume)

    def _handle_resume(self, resume_arg: Union[bool, str]):
        """Restores state from checkpoint if available."""
        # Check if Google Drive has a backup to restore if local is empty
        self.drive_manager.restore_if_available(self.output_dir)

        if isinstance(resume_arg, str) and resume_arg != "auto":
            ckpt_path = resume_arg
        else:
            ckpt_path = self.checkpoint.latest_checkpoint_path

        if os.path.exists(ckpt_path):
            try:
                loaded_epoch, best_metric, history, extra_state = self.checkpoint.load(
                    checkpoint_path=ckpt_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    device=self.device,
                )
                self.start_epoch = loaded_epoch
                self.best_acc1 = best_metric
                self.logger.set_history(history)

                # Restore TransferGuard dynamic routing state
                if self.is_transferguard and isinstance(self.model, AdaptiveRoutingModel):
                    if "running_scores" in extra_state:
                        self.model.running_scores = extra_state["running_scores"]
                    if "history_scores" in extra_state:
                        self.model.history_scores = extra_state["history_scores"]
                    if "routing_actions" in extra_state:
                        stages_to_freeze = [
                            k for k, act in extra_state["routing_actions"].items() if act == "freeze"
                        ]
                        self.model.backbone.freeze_stages(stages_to_freeze)

                print("\n" + "=" * 65)
                print(f"🔄 RESUMED TRAINING CHECKPOINT:")
                print(f"   Source File:       {ckpt_path}")
                print(f"   Completed Epochs:  {self.start_epoch}")
                print(f"   Best Prior Acc:    {self.best_acc1:.2f}%")
                print("=" * 65)
            except Exception as e:
                print(f"⚠️ Warning: Failed to resume checkpoint from {ckpt_path}: {e}")
                print("   Proceeding with initial weights.")
        else:
            if resume_arg != "auto" and isinstance(resume_arg, str):
                raise FileNotFoundError(f"Specified resume checkpoint does not exist: {ckpt_path}")
            print(f"ℹ️ No prior checkpoint found at {ckpt_path}. Starting clean run from Epoch 1.")

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_acc1 = 0.0
        num_batches = len(dataloader)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if self.is_transferguard and isinstance(self.model, AdaptiveRoutingModel):
                    logits, gate_scores = self.model(images)
                    loss_dict = self.criterion(
                        logits=logits,
                        targets=targets,
                        gate_scores=gate_scores,
                        controller=self.model.controller,
                    )
                    self.model.record_batch_scores(gate_scores)
                else:
                    logits = self.model(images)
                    loss_dict = self.criterion(logits=logits, targets=targets)

                loss = loss_dict["total_loss"]

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            acc1 = accuracy(logits, targets, topk=(1,))[0]
            total_loss += loss.item()
            total_acc1 += acc1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc1:.2f}%"})

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "train_loss": total_loss / num_batches,
            "train_acc1": total_acc1 / num_batches,
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    if self.is_transferguard and isinstance(self.model, AdaptiveRoutingModel):
                        logits, _ = self.model(images)
                    else:
                        logits = self.model(images)

                    loss_dict = self.criterion(logits=logits, targets=targets)
                acc1, acc5 = accuracy(
                    logits, targets, topk=(1, min(5, logits.shape[1]))
                )

                total_loss += loss_dict["total_loss"].item()
                total_acc1 += acc1
                total_acc5 += acc5

        return {
            "val_loss": total_loss / num_batches,
            "val_acc1": total_acc1 / num_batches,
            "val_acc5": total_acc5 / num_batches,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
    ) -> Dict[str, Any]:
        """Runs or resumes training workflow across all epochs."""
        history_path = os.path.join(self.output_dir, "train_history.json")

        if self.start_epoch >= epochs:
            print(f"\n✅ Training already completed for {epochs} epochs (current: {self.start_epoch}).")
            val_metrics = self.evaluate(val_loader)
            print(f"   Validation Top-1 Accuracy: {val_metrics['val_acc1']:.2f}%")
            return {
                "best_val_acc1": self.best_acc1 or val_metrics["val_acc1"],
                "total_time": 0.0,
                "history_file": history_path,
            }

        start_time = time.time()
        start_ep = self.start_epoch + 1

        print(f"\n🚀 Commencing Training (Epochs {start_ep} -> {epochs}) on device: {self.device}")
        print("=" * 65)

        for epoch in range(start_ep, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)

            # TransferGuard dynamic routing update
            routing_actions = {}
            extra_state = {}
            if self.is_transferguard and isinstance(self.model, AdaptiveRoutingModel):
                routing_actions = self.model.update_dynamic_routing()
                extra_state = {
                    "running_scores": dict(self.model.running_scores),
                    "history_scores": list(self.model.history_scores),
                    "routing_actions": routing_actions,
                }

            combined_metrics = {**train_metrics, **val_metrics}
            self.logger.log_epoch(epoch, combined_metrics)

            is_best = self.checkpoint.step(
                current_metric=val_metrics["val_acc1"],
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                history=self.logger.history,
                extra_state=extra_state,
            )

            if val_metrics["val_acc1"] > self.best_acc1:
                self.best_acc1 = val_metrics["val_acc1"]

            # Save metrics history immediately after every epoch so progress is never lost
            self.logger.save_history(history_path)
            self.start_epoch = epoch

            # Sync checkpoints to persistent Google Drive if enabled
            self.drive_manager.sync(self.output_dir)

            # Formatted epoch printout
            status = f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_metrics['train_loss']:.4f} | Train Acc: {train_metrics['train_acc1']:.2f}% | Val Acc: {val_metrics['val_acc1']:.2f}%"
            if is_best:
                status += " ⭐ [Best]"
            print(status)

            if routing_actions:
                action_str = ", ".join([f"{k}: {v}" for k, v in routing_actions.items()])
                print(f"   ↳ TransferGuard Routing: [{action_str}]")

        total_time = time.time() - start_time
        print("=" * 65)
        print(f"✨ Training session complete in {total_time / 60:.2f} mins. Best Val Acc: {self.best_acc1:.2f}%")

        return {
            "best_val_acc1": self.best_acc1,
            "total_time": total_time,
            "history_file": history_path,
        }
