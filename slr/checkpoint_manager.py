"""
Checkpoint Manager
==================
Save and resume training state — model weights, optimiser, scheduler,
epoch number, best metric, and training history.

Usage:
    ckpt = CheckpointManager("checkpoints/google_asl")

    # Inside training loop:
    ckpt.save(epoch, model, optimizer, scheduler, val_f1, is_best=val_f1 > best_f1)

    # To resume:
    start_epoch, best_f1 = ckpt.load_latest(model, optimizer, scheduler)
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Manages saving and loading of training checkpoints.

    Saves two checkpoint files:
      - latest.pt   : always the most recent epoch (for resuming)
      - best.pt     : the epoch with the highest validation metric

    Also maintains a human-readable training_log.json for quick inspection.

    Args:
        save_dir     : directory to write checkpoints into (created if missing).
        metric_name  : name of the tracked metric, used in log output.
        verbose      : print a message on every save / load.
    """

    LATEST_NAME = "latest.pt"
    BEST_NAME   = "best.pt"
    LOG_NAME    = "training_log.json"

    def __init__(
        self,
        save_dir: str,
        metric_name: str = "val_f1",
        verbose: bool = True,
    ):
        self.save_dir    = Path(save_dir)
        self.metric_name = metric_name
        self.verbose     = verbose
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._log: list[dict] = self._load_log()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        metric_value: float,
        is_best: bool,
        extra: Optional[dict] = None,
    ) -> None:
        """
        Save a checkpoint for the current epoch.

        Args:
            epoch        : current epoch number (0-indexed).
            model        : the model to save.
            optimizer    : the optimizer state.
            scheduler    : the LR scheduler state.
            metric_value : value of the tracked validation metric.
            is_best      : if True, also copies this checkpoint to best.pt.
            extra        : any additional keys to store (e.g. label_map).
        """
        state = {
            "epoch"             : epoch,
            "model_state_dict"  : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            self.metric_name    : metric_value,
            "extra"             : extra or {},
        }

        latest_path = self.save_dir / self.LATEST_NAME
        torch.save(state, latest_path)

        if is_best:
            best_path = self.save_dir / self.BEST_NAME
            shutil.copyfile(latest_path, best_path)
            if self.verbose:
                print(f"  ✦ New best checkpoint  — epoch {epoch:3d}  {self.metric_name}={metric_value:.4f}")

        # Append to training log
        log_entry = {"epoch": epoch, self.metric_name: metric_value, "is_best": is_best}
        if extra:
            log_entry.update({k: v for k, v in extra.items() if _json_serialisable(v)})
        self._log.append(log_entry)
        self._save_log()

        if self.verbose and not is_best:
            print(f"  · Checkpoint saved     — epoch {epoch:3d}  {self.metric_name}={metric_value:.4f}")

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        device: Optional[torch.device] = None,
    ) -> tuple[int, float]:
        """
        Load the most recent checkpoint (latest.pt) to resume training.

        Args:
            model     : model to load weights into.
            optimizer : optimizer to restore (optional — skip if changing LR).
            scheduler : LR scheduler to restore (optional).
            device    : map_location device.

        Returns:
            (start_epoch, best_metric_value)
            start_epoch is epoch + 1 (i.e. the next epoch to run).
        """
        return self._load(
            self.save_dir / self.LATEST_NAME,
            model, optimizer, scheduler, device,
        )

    def load_best(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> float:
        """
        Load the best checkpoint (best.pt) for evaluation / fine-tuning.
        Optimizer and scheduler states are not restored.

        Returns:
            best metric value stored in the checkpoint.
        """
        epoch, metric = self._load(
            self.save_dir / self.BEST_NAME,
            model, optimizer=None, scheduler=None, device=device,
        )
        return metric

    def has_checkpoint(self, which: str = "latest") -> bool:
        """Check whether a checkpoint file exists."""
        fname = self.LATEST_NAME if which == "latest" else self.BEST_NAME
        return (self.save_dir / fname).exists()

    def get_log(self) -> list[dict]:
        """Return the full training history log."""
        return self._log

    def best_metric(self) -> Optional[float]:
        """Return the best metric value seen so far (from the log)."""
        bests = [e[self.metric_name] for e in self._log if e.get("is_best")]
        return max(bests) if bests else None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, path: Path, model, optimizer, scheduler, device):
        if not path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {path}.\n"
                "Start training from scratch or check the save_dir path."
            )
        map_loc = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state   = torch.load(path, map_location=map_loc, weights_only=False)

        model.load_state_dict(state["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        if scheduler is not None and state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        epoch  = state["epoch"]
        metric = state.get(self.metric_name, 0.0)

        if self.verbose:
            print(
                f"  ✔ Loaded checkpoint '{path.name}' — "
                f"epoch={epoch}  {self.metric_name}={metric:.4f}"
            )
        return epoch + 1, metric   # next epoch to run

    def _load_log(self) -> list:
        log_path = self.save_dir / self.LOG_NAME
        if log_path.exists():
            with open(log_path) as f:
                return json.load(f)
        return []

    def _save_log(self) -> None:
        with open(self.save_dir / self.LOG_NAME, "w") as f:
            json.dump(self._log, f, indent=2)


# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stops training when a monitored metric stops improving.

    The paper uses patience=20 on validation loss.

    Args:
        patience  : epochs to wait after last improvement.
        min_delta : minimum change to count as improvement.
        mode      : 'min' (loss) or 'max' (accuracy / F1).
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = "min"):
        assert mode in ("min", "max")
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best       = float("inf") if mode == "min" else float("-inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """
        Call once per epoch with the latest metric.

        Returns:
            True if training should stop, False otherwise.
        """
        improved = (
            metric < self.best - self.min_delta
            if self.mode == "min"
            else metric > self.best + self.min_delta
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        self.best        = float("inf") if self.mode == "min" else float("-inf")
        self.counter     = 0
        self.should_stop = False


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _json_serialisable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False
