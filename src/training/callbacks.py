import os
import json
import torch
from typing import Any, Dict, List, Optional, Tuple, Union


class MetricLogger:
    """Tracks running training metrics and maintains historical records."""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        record = {"epoch": epoch, **metrics}
        self.history.append(record)

    def set_history(self, history: List[Dict[str, Any]]):
        """Sets historical records when resuming from a checkpoint."""
        self.history = list(history)

    def save_history(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_history(self, filepath: str) -> List[Dict[str, Any]]:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self.history = json.load(f)
        return self.history


class ModelCheckpoint:
    """Saves and restores model weights, optimizer/scheduler state, and metrics."""

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
        scheduler: Optional[Any] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Saves latest checkpoint. If current_metric beats best, updates best checkpoint.
        Returns True if a new best was achieved.
        """
        is_best = False
        if (self.mode == "max" and current_metric > self.best_metric) or (
            self.mode == "min" and current_metric < self.best_metric
        ):
            self.best_metric = current_metric
            is_best = True

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metric": current_metric,
            "best_metric": self.best_metric,
            "history": history or [],
            "extra_state": extra_state or {},
        }

        # Save latest
        torch.save(state, self.latest_checkpoint_path)

        # Save best if new high score
        if is_best:
            torch.save(state, self.best_checkpoint_path)

        return is_best

    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[int, float, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Restores checkpoint state onto model, optimizer, and scheduler.
        Returns (start_epoch, best_metric, history, extra_state).
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Restore weights
        model.load_state_dict(ckpt["model_state_dict"])

        # Restore optimizer
        if optimizer is not None and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Restore scheduler
        if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        epoch = ckpt.get("epoch", 0)
        best_metric = ckpt.get("best_metric", ckpt.get("metric", 0.0))
        self.best_metric = best_metric
        history = ckpt.get("history", [])
        extra_state = ckpt.get("extra_state", {})

        return epoch, best_metric, history, extra_state
