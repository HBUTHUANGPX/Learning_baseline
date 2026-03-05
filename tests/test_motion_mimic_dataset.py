"""Minimal tests for motion-mimic dataset integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import argparse

from modules.data import DataConfig, create_dataloader
from scripts.train_vae import ExperimentManager


def _write_motion_npz(path: Path, length: int = 12, joints: int = 6, bodies: int = 4) -> None:
    """Writes one synthetic motion NPZ for testing.

    Args:
        path: Output NPZ path.
        length: Sequence length.
        joints: Number of joints.
        bodies: Number of bodies.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    np.savez(
        path,
        fps=np.array(30.0, dtype=np.float32),
        joint_pos=rng.normal(size=(length, joints)).astype(np.float32),
        joint_vel=rng.normal(size=(length, joints)).astype(np.float32),
        body_pos_w=rng.normal(size=(length, bodies, 3)).astype(np.float32),
    )


def test_motion_mimic_sequence_batch_protocol(tmp_path: Path) -> None:
    """Tests motion dataset in sequence mode with protocol batch output."""
    file_a = tmp_path / "motion_a.npz"
    file_b = tmp_path / "motion_b.npz"
    _write_motion_npz(file_a, length=10)
    _write_motion_npz(file_b, length=8)

    config = DataConfig(
        dataset="motion_mimic",
        batch_size=2,
        val_ratio=0.0,
        motion_files=(str(file_a), str(file_b)),
        motion_feature_keys=("joint_pos", "joint_vel"),
        motion_as_sequence=True,
        motion_frame_stride=1,
        use_batch_protocol=True,
    )
    train_loader, _ = create_dataloader(config)
    batch = next(iter(train_loader))

    assert "obs" in batch
    assert "policy" in batch["obs"]
    assert batch["obs"]["policy"].ndim == 3
    assert "policy_lengths" in batch["meta"]
    assert "policy_mask" in batch["meta"]


def test_motion_mimic_frame_mode_batch_protocol(tmp_path: Path) -> None:
    """Tests motion dataset in frame mode with state-shaped policy tensor."""
    file_a = tmp_path / "motion_frame.npz"
    _write_motion_npz(file_a, length=9)

    config = DataConfig(
        dataset="motion_mimic",
        batch_size=4,
        val_ratio=0.0,
        motion_files=(str(file_a),),
        motion_feature_keys=("joint_pos", "joint_vel"),
        motion_as_sequence=False,
        use_batch_protocol=True,
    )
    train_loader, _ = create_dataloader(config)
    batch = next(iter(train_loader))

    assert batch["obs"]["policy"].ndim == 2
    assert batch["obs"]["policy"].shape[0] == 4


def test_motion_mimic_auto_infers_input_dim_for_mlp(tmp_path: Path) -> None:
    """Tests ExperimentManager infers MLP input_dim from motion feature keys."""
    file_a = tmp_path / "motion_auto_dim.npz"
    _write_motion_npz(file_a, length=10, joints=7)
    # joint_pos + joint_vel -> 14 feature dims per frame
    args = argparse.Namespace(
        algorithm="vae",
        model="vanilla",
        dataset="motion_mimic",
        input_dim=784,
        image_channels=1,
        image_height=28,
        image_width=28,
        latent_dim=4,
        hidden_dims="16,8",
        conv_channels="32,64",
        conv_bottleneck_dim=256,
        vq_decoder_channels="64,32",
        activation="relu",
        beta=4.0,
        num_embeddings=32,
        fsq_levels=6,
        epochs=1,
        batch_size=4,
        num_samples=8,
        sequence_length=32,
        sequence_feature_dim=16,
        sequence_variable_length=False,
        sequence_min_length=8,
        motion_files=(str(file_a),),
        motion_file_yaml="",
        motion_group="",
        motion_feature_keys=("joint_pos", "joint_vel"),
        motion_as_sequence=False,
        motion_frame_stride=1,
        motion_normalize=False,
        no_batch_protocol=False,
        lr=1e-3,
        seed=42,
        deterministic=False,
        device="cpu",
        data_root=str(tmp_path / "data"),
        log_root=str(tmp_path / "log"),
    )
    manager = ExperimentManager(args)
    assert manager.args.input_dim == 14
    manager.tb_logger.close()
