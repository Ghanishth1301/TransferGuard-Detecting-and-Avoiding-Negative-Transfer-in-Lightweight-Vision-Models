import os
import time
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union
from torch.utils.data import DataLoader
from tqdm import tqdm
from .loss import TransferGuardLoss
from .callbacks import MetricLogger, ModelCheckpoint
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
    linear probing, and TransferGuard adaptive routing.
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

        os.makedirs(output_dir, exist_ok=True)
        self.logger = MetricLogger()
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=os.path.join(output_dir, "checkpoints")
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_acc1 = 0.0
        num_batches = len(dataloader)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

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
            loss.backward()
            self.optimizer.step()

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
                images = images.to(self.device)
                targets = targets.to(self.device)

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
        """Runs complete training workflow across all epochs."""
        start_time = time.time()
        best_acc1 = 0.0

        print(f"\n🚀 Commencing Training for {epochs} epochs on device: {self.device}")
        print("=" * 65)

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)

            # TransferGuard dynamic routing update
            routing_actions = {}
            if self.is_transferguard and isinstance(self.model, AdaptiveRoutingModel):
                routing_actions = self.model.update_dynamic_routing()

            combined_metrics = {**train_metrics, **val_metrics}
            self.logger.log_epoch(epoch, combined_metrics)

            is_best = self.checkpoint.step(
                current_metric=val_metrics["val_acc1"],
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                extra_state={"routing_actions": routing_actions},
            )

            if val_metrics["val_acc1"] > best_acc1:
                best_acc1 = val_metrics["val_acc1"]

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
        print(f"✨ Training complete in {total_time / 60:.2f} mins. Best Val Acc: {best_acc1:.2f}%")

        # Save training history JSON
        history_path = os.path.join(self.output_dir, "train_history.json")
        self.logger.save_history(history_path)

        return {
            "best_val_acc1": best_acc1,
            "total_time": total_time,
            "history_file": history_path,
        }

