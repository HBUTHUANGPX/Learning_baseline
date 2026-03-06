"""Minimal smoke tests for context-conditioned motion VQ/FSQ training loop."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from scripts.train_motion_vqvae import _build_model, main


def _write_motion_npz(path: Path, length: int = 20, joints: int = 6) -> None:
    """Writes synthetic motion data for training smoke tests.

    Args:
        path: Target NPZ path.
        length: Number of frames.
        joints: Number of joint dimensions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    np.savez(
        path,
        fps=np.array(30.0, dtype=np.float32),
        joint_pos=rng.normal(size=(length, joints)).astype(np.float32),
        joint_vel=rng.normal(size=(length, joints)).astype(np.float32),
    )


def test_train_motion_vqvae_one_epoch(tmp_path: Path) -> None:
    """Tests one epoch of context-conditioned FSQ training end-to-end."""
    file_a = tmp_path / "motion.npz"
    _write_motion_npz(file_a)

    args = Namespace(
        model="fsq",
        embedding_dim=8,
        hidden_dim=32,
        num_embeddings=16,
        fsq_levels=6,
        beta=0.25,
        recon_loss_mode="mse",
        batch_size=8,
        epochs=1,
        lr=1e-3,
        seed=42,
        deterministic=False,
        device="cpu",
        motion_files=str(file_a),
        motion_file_yaml="",
        motion_group="",
        motion_feature_keys="joint_pos,joint_vel",
        motion_frame_stride=1,
        motion_normalize=False,
        history_frames=2,
        future_frames=1,
        val_ratio=0.2,
        log_root=str(tmp_path / "log"),
    )
    main(args)

    checkpoints = list((tmp_path / "log").glob("*/checkpoint/epoch_001.pt"))
    assert len(checkpoints) == 1


def test_build_model_supports_conditioned_decoder() -> None:
    """Tests model factory supports conditioned decoder dimensions."""
    args = Namespace(
        model="vq",
        embedding_dim=8,
        hidden_dim=32,
        num_embeddings=16,
        fsq_levels=6,
        beta=0.25,
        recon_loss_mode="mse",
    )
    model = _build_model(
        args,
        encoder_input_dim=24,
        decoder_condition_dim=12,
        target_dim=18,
    )
    enc = torch.rand(2, 24)
    cond = torch.rand(2, 12)
    target = torch.rand(2, 18)
    outputs = model(enc, cond)
    losses = model.loss_function(target, outputs)
    assert outputs["x_hat"].shape == (2, 18)
    assert losses["loss"].dim() == 0
