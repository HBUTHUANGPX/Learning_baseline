"""Minimal tests for context-conditioned motion dataloader pipeline."""

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


def test_create_motion_dataloaders_no_context() -> None:
    """Tests default no-context dimensions and batch keys."""
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
    (
        train_loader,
        val_loader,
        encoder_input_dim,
        decoder_condition_dim,
        target_dim,
    ) = create_motion_dataloaders(config)

    assert encoder_input_dim == 10
    assert decoder_condition_dim == 0
    assert target_dim == 10
    batch = next(iter(train_loader))
    assert batch["encoder_input"].shape[1] == 10
    assert batch["decoder_condition"].shape[1] == 0
    assert batch["target"].shape[1] == 10
    assert len(train_loader.dataset) + len(val_loader.dataset) == 16


def test_context_window_condition_and_target_shape() -> None:
    """Tests context-conditioned shape contract: enc=h+cur+f, dec_cond=h, target=cur+f."""
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
    )
    train_loader, _, enc_dim, cond_dim, target_dim = create_motion_dataloaders(config)
    # frame_dim = 8
    # encoder window = 2 + 1 + 3 = 6 -> 48
    # condition window = 2 -> 16
    # target window = 1 + 3 = 4 -> 32
    assert enc_dim == 48
    assert cond_dim == 16
    assert target_dim == 32

    batch = next(iter(train_loader))
    assert batch["encoder_input"].shape[1] == 48
    assert batch["decoder_condition"].shape[1] == 16
    assert batch["target"].shape[1] == 32
