"""TensorBoard and experiment-path helpers for training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

from torch.utils.tensorboard import SummaryWriter


@dataclass
class ExperimentPaths:
    """Container for experiment directory structure.

    Attributes:
        root: Log root directory.
        run_dir: Timestamped run directory.
        tensorboard_dir: TensorBoard event directory.
        checkpoint_dir: Checkpoint output directory.
    """

    root: Path
    run_dir: Path
    tensorboard_dir: Path
    checkpoint_dir: Path


def create_experiment_paths(log_root: str = "./log") -> ExperimentPaths:
    """Creates timestamped experiment output directories.

    Args:
        log_root: Root directory for all runs.

    Returns:
        Experiment path container with created directories.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(log_root)
    run_dir = root / run_id
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir = run_dir / "checkpoint"

    for path in (root, run_dir, tensorboard_dir, checkpoint_dir):
        path.mkdir(parents=True, exist_ok=True)

    return ExperimentPaths(
        root=root,
        run_dir=run_dir,
        tensorboard_dir=tensorboard_dir,
        checkpoint_dir=checkpoint_dir,
    )


class TensorboardLogger:
    """Thin wrapper around ``SummaryWriter`` for scalar metrics."""

    def __init__(self, log_dir: Path) -> None:
        """Initializes TensorBoard writer.

        Args:
            log_dir: Event file output directory.
        """
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str) -> None:
        """Logs scalar metrics under a prefix.

        Args:
            metrics: Name-value metric dictionary.
            step: Global step value.
            prefix: Namespace prefix, for example ``"train"``.
        """
        for key, value in metrics.items():
            self._writer.add_scalar(f"{prefix}/{key}", value, step)

    def close(self) -> None:
        """Flushes and closes the writer."""
        self._writer.close()
