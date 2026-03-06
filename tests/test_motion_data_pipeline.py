"""Minimal tests for context-aware motion dataloader pipeline."""

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


def test_create_motion_dataloaders_current_target() -> None:
    """Tests dataloader creation with default current-frame targets."""
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
    train_loader, val_loader, input_dim, target_dim = create_motion_dataloaders(config)

    assert input_dim == 10
    assert target_dim == 10
    batch = next(iter(train_loader))
    assert batch["input"].ndim == 2
    assert batch["input"].shape[1] == 10
    assert batch["target"].shape[1] == 10
    assert len(train_loader.dataset) + len(val_loader.dataset) == 16


def test_context_window_and_all_target_shape() -> None:
    """Tests context-window indexing and all-frame reconstruction target."""
    root = Path("/tmp/motion_data_pipeline_context")
    file_a = root / "a.npz"
    _write_motion_npz(file_a, length=12, joints=4)

    config = DataConfig(
        batch_size=2,
        val_ratio=0.0,
        motion_files=(str(file_a),),
        motion_feature_keys=("joint_pos", "joint_vel"),
        history_frames=2,
        future_frames=3,
        reconstruction_target="all",
    )
    train_loader, _, input_dim, target_dim = create_motion_dataloaders(config)
    # frame_dim = 8, window_size = 6 -> flattened dim = 48
    assert input_dim == 48
    assert target_dim == 48
    batch = next(iter(train_loader))
    assert batch["input"].shape[1] == 48
    assert batch["target"].shape[1] == 48


def test_future_target_shape() -> None:
    """Tests future-target mode with valid offset in context window."""
    root = Path("/tmp/motion_data_pipeline_future")
    file_a = root / "a.npz"
    _write_motion_npz(file_a, length=12, joints=3)

    config = DataConfig(
        batch_size=2,
        val_ratio=0.0,
        motion_files=(str(file_a),),
        motion_feature_keys=("joint_pos", "joint_vel"),
        history_frames=1,
        future_frames=2,
        reconstruction_target="future",
        future_target_offset=2,
    )
    train_loader, _, input_dim, target_dim = create_motion_dataloaders(config)
    # frame_dim = 6, window_size = 4 -> input 24, future target 6
    assert input_dim == 24
    assert target_dim == 6
    batch = next(iter(train_loader))
    assert batch["input"].shape[1] == 24
    assert batch["target"].shape[1] == 6
