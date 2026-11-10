import os
import json
import torch
from typing import Any, Dict, List, Optional


class MetricLogger:
    """Tracks running training metrics and maintains historical records."""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        record = {"epoch": epoch, **metrics}
        self.history.append(record)

    def save_history(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)


class ModelCheckpoint:
    """Saves best model weights and training checkpoints."""

    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = "val_acc1",
        mode: str = "max",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.best_metric = -float("inf") if mode == "max" else float("inf")
        self.best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
        self.latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_model.pth")
        os.makedirs(checkpoint_dir, exist_ok=True)

    def step(
        self,
        current_metric: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Saves latest checkpoint. If current_metric beats best, updates best checkpoint.
        Returns True if a new best was achieved.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric": current_metric,
            "extra_state": extra_state or {},
        }
        # Save latest
        torch.save(state, self.latest_checkpoint_path)

        is_best = False
        if (self.mode == "max" and current_metric > self.best_metric) or (
            self.mode == "min" and current_metric < self.best_metric
        ):
            self.best_metric = current_metric
            torch.save(state, self.best_checkpoint_path)
            is_best = True

        return is_best

