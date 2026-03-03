"""TensorBoard and experiment directory utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter


@dataclass
class ExperimentPaths:
    """Container for paths used by one experiment run."""

    root: Path
    run_dir: Path
    tensorboard_dir: Path
    checkpoint_dir: Path
    reconstructions_dir: Path


def create_experiment_paths(log_root: str = "./log", timestamp: str | None = None) -> ExperimentPaths:
    """Creates experiment directories using timestamp-based run IDs.

    Args:
        log_root: Root directory for all experiment logs.
        timestamp: Optional timestamp string. If ``None``, current time is used.

    Returns:
        ``ExperimentPaths`` with created folders.
    """
    run_id = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(log_root)
    run_dir = root / run_id
    tb_dir = run_dir / "tensorboard"
    ckpt_dir = run_dir / "checkpoint"
    recon_dir = run_dir / "reconstructions"

    for directory in (root, run_dir, tb_dir, ckpt_dir, recon_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return ExperimentPaths(
        root=root,
        run_dir=run_dir,
        tensorboard_dir=tb_dir,
        checkpoint_dir=ckpt_dir,
        reconstructions_dir=recon_dir,
    )


class TensorboardLogger:
    """Thin wrapper around ``SummaryWriter`` for structured scalar logging."""

    def __init__(self, log_dir: Path) -> None:
        """Initializes TensorBoard writer.

        Args:
            log_dir: Directory where event files are saved.
        """
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalars(self, scalars: Dict[str, float], step: int, prefix: str) -> None:
        """Logs a scalar dictionary under a namespace prefix.

        Args:
            scalars: Mapping from metric names to scalar values.
            step: Global training step or epoch.
            prefix: Namespace prefix such as ``"train"`` or ``"val"``.
        """
        for key, value in scalars.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def log_reconstruction(self, x: torch.Tensor, x_hat: torch.Tensor, step: int) -> None:
        """Logs reconstruction tensors to TensorBoard.

        Args:
            x: Original inputs, either flattened or image tensors.
            x_hat: Reconstructed outputs.
            step: Training step or epoch.
        """
        original = x[:8]
        reconstruction = x_hat[:8]
        if original.ndim == 4:
            self.writer.add_images("reconstruction/original", original, step)
            self.writer.add_images("reconstruction/prediction", reconstruction, step)
            return
        # Fallback for flattened vectors.
        self.writer.add_histogram("reconstruction/original", original, step)
        self.writer.add_histogram("reconstruction/prediction", reconstruction, step)

    def close(self) -> None:
        """Closes the underlying writer and flushes pending events."""
        self.writer.close()
