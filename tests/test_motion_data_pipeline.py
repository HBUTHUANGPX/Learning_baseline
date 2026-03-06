"""Minimal tests for motion frame dataloader pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modules.data import DataConfig, create_motion_dataloaders


def _write_motion_npz(path: Path, length: int = 12, joints: int = 6) -> None:
    """Writes a synthetic motion NPZ file for tests.

    Args:
        path: Target NPZ path.
        length: Number of frames.
        joints: Number of joint dimensions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    np.savez(
        path,
        fps=np.array(30.0, dtype=np.float32),
        joint_pos=rng.normal(size=(length, joints)).astype(np.float32),
        joint_vel=rng.normal(size=(length, joints)).astype(np.float32),
    )


def test_create_motion_dataloaders_frame_level() -> None:
    """Tests train/validation dataloader creation and input dim inference."""
    root = Path("/tmp/motion_data_pipeline")
    file_a = root / "a.npz"
    file_b = root / "b.npz"
    _write_motion_npz(file_a, length=10, joints=5)
    _write_motion_npz(file_b, length=6, joints=5)

    config = DataConfig(
        batch_size=4,
        val_ratio=0.25,
        motion_files=(str(file_a), str(file_b)),
        motion_feature_keys=("joint_pos", "joint_vel"),
    )
    train_loader, val_loader, input_dim = create_motion_dataloaders(config)

    assert input_dim == 10
    batch = next(iter(train_loader))
    assert batch["state"].ndim == 2
    assert batch["state"].shape[1] == 10
    assert len(train_loader.dataset) + len(val_loader.dataset) == 16
