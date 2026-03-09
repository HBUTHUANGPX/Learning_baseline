"""Tests for latent-condition data pipeline.

Coverage target: > 85% for this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from modules.latent_data import (
    FrozenFSQLatentEncoder,
    LatentDataConfig,
    create_latent_condition_loaders,
)
from modules.vqvae import FrameFSQVAE


def _write_motion_npz(path: Path, length: int = 20, joints: int = 2) -> None:
    """Writes minimal motion NPZ for dataset tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    np.savez(
        path,
        fps=np.array(30.0, dtype=np.float32),
        joint_pos=rng.normal(size=(length, joints)).astype(np.float32),
        joint_vel=rng.normal(size=(length, joints)).astype(np.float32),
    )


def _write_fsq_checkpoint(path: Path) -> None:
    """Writes a minimal FSQ checkpoint compatible with latent loader."""
    model = FrameFSQVAE(
        encoder_input_dim=44,  # (2 + 1 + 8) * 4
        decoder_condition_dim=8,  # 2 * 4
        target_dim=36,  # (1 + 8) * 4
        embedding_dim=6,
        hidden_dim=16,
        fsq_levels=8,
        recon_loss_mode="mse",
    )
    state = {
        "encoder_input_dim": 44,
        "decoder_condition_dim": 8,
        "target_dim": 36,
        "args": {
            "embedding_dim": 6,
            "hidden_dim": 16,
            "fsq_levels": 8,
            "recon_loss_mode": "mse",
        },
        "model_state": model.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _write_text_token_pt(path: Path, npz_name: str, text_dim: int = 16) -> None:
    """Writes minimal trajectory text token mapping."""
    mapping = {
        npz_name: {
            "clip_text_prompt": "A person walks forward.",
            "pooler_output": torch.randn(text_dim),
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mapping, path)


def test_create_latent_condition_loaders_shapes(tmp_path: Path) -> None:
    """Tests latent loader output shapes for minimal valid setup."""
    motion_path = tmp_path / "a.npz"
    ckpt_path = tmp_path / "fsq.pt"
    text_path = tmp_path / "text.pt"
    _write_motion_npz(motion_path, length=24, joints=2)
    _write_fsq_checkpoint(ckpt_path)
    _write_text_token_pt(text_path, npz_name=motion_path.name, text_dim=16)

    cfg = LatentDataConfig(
        batch_size=4,
        val_ratio=0.0,
        motion_files=(str(motion_path),),
        motion_feature_keys=("joint_pos", "joint_vel"),
        history_frames=2,
        future_frames=8,
        text_token_pt=str(text_path),
        fsq_checkpoint=str(ckpt_path),
        fsq_device="cpu",
        motion_cache_device="cpu",
    )
    train_loader, _, meta = create_latent_condition_loaders(cfg)
    batch = next(iter(train_loader))

    assert meta["cond_motion_dim"] == 8
    assert batch["cond_motion"].shape[1] == meta["cond_motion_dim"]
    assert batch["cond_text"].shape[1] == meta["text_dim"]
    assert batch["target_latent"].shape[1] == meta["latent_dim"]
    assert_close(batch["cond_motion"].isfinite().float().mean(), torch.tensor(1.0))
    assert_close(batch["target_latent"].isfinite().float().mean(), torch.tensor(1.0))


def test_create_latent_condition_loaders_missing_text_key_raises(tmp_path: Path) -> None:
    """Tests loader raises when text mapping misses required NPZ key."""
    motion_path = tmp_path / "a.npz"
    ckpt_path = tmp_path / "fsq.pt"
    text_path = tmp_path / "text.pt"
    _write_motion_npz(motion_path, length=24, joints=2)
    _write_fsq_checkpoint(ckpt_path)
    _write_text_token_pt(text_path, npz_name="wrong_key.npz", text_dim=16)

    cfg = LatentDataConfig(
        batch_size=2,
        val_ratio=0.0,
        motion_files=(str(motion_path),),
        motion_feature_keys=("joint_pos", "joint_vel"),
        history_frames=2,
        future_frames=8,
        text_token_pt=str(text_path),
        fsq_checkpoint=str(ckpt_path),
        fsq_device="cpu",
        motion_cache_device="cpu",
    )
    with pytest.raises(KeyError):
        _ = create_latent_condition_loaders(cfg)


def test_frozen_fsq_latent_encoder_dim_mismatch_raises(tmp_path: Path) -> None:
    """Tests frozen FSQ latent encoder validates input dimension."""
    ckpt_path = tmp_path / "fsq.pt"
    _write_fsq_checkpoint(ckpt_path)
    encoder = FrozenFSQLatentEncoder(str(ckpt_path), device="cpu")
    wrong_input = torch.randn(2, encoder.encoder_input_dim + 1)
    with pytest.raises(ValueError):
        _ = encoder.encode_z_q(wrong_input)


def test_create_latent_condition_loaders_invalid_n_gt_1_plus_m_raises(tmp_path: Path) -> None:
    """Tests loader rejects invalid AR setting where n > 1 + m."""
    motion_path = tmp_path / "a.npz"
    ckpt_path = tmp_path / "fsq.pt"
    text_path = tmp_path / "text.pt"
    _write_motion_npz(motion_path, length=24, joints=2)
    _write_fsq_checkpoint(ckpt_path)
    _write_text_token_pt(text_path, npz_name=motion_path.name, text_dim=16)

    cfg = LatentDataConfig(
        batch_size=2,
        val_ratio=0.0,
        motion_files=(str(motion_path),),
        motion_feature_keys=("joint_pos", "joint_vel"),
        history_frames=10,
        future_frames=8,
        text_token_pt=str(text_path),
        fsq_checkpoint=str(ckpt_path),
        fsq_device="cpu",
        motion_cache_device="cpu",
    )
    with pytest.raises(ValueError, match="history_frames <= 1 \\+ future_frames"):
        _ = create_latent_condition_loaders(cfg)
