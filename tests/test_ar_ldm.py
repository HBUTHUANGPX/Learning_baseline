"""Tests for AR-LDM model module.

Coverage target: > 85% for this module.
"""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from modules.ar_ldm import ARLDMConfig, ARLDMTransformer


def _build_batch(
    batch_size: int = 4,
    cond_motion_dim: int = 12,
    text_dim: int = 16,
    latent_dim: int = 6,
) -> dict[str, torch.Tensor]:
    """Builds minimal synthetic batch for AR-LDM tests."""
    return {
        "cond_motion": torch.randn(batch_size, cond_motion_dim),
        "cond_text": torch.randn(batch_size, text_dim),
        "target_latent": torch.randn(batch_size, latent_dim),
    }


@pytest.mark.parametrize("fusion_type", ["concat", "cross_attention"])
def test_ar_ldm_forward_shape_transformer(fusion_type: str) -> None:
    """Tests forward output shape for transformer backbone with both fusion modes."""
    cfg = ARLDMConfig(
        latent_dim=6,
        cond_motion_dim=12,
        text_dim=16,
        model_dim=32,
        num_layers=2,
        num_heads=4,
        timesteps=20,
        backbone_type="transformer",
        fusion_type=fusion_type,
    )
    model = ARLDMTransformer(cfg)
    batch = _build_batch()
    t = torch.randint(0, cfg.timesteps, (batch["target_latent"].shape[0],), dtype=torch.long)
    out = model(
        x_t=batch["target_latent"],
        timesteps=t,
        cond_motion=batch["cond_motion"],
        cond_text=batch["cond_text"],
    )
    assert out.shape == batch["target_latent"].shape


def test_ar_ldm_training_step_backward_mlp() -> None:
    """Tests MLP-backbone training step is numerically stable and backpropagates."""
    cfg = ARLDMConfig(
        latent_dim=6,
        cond_motion_dim=12,
        text_dim=16,
        model_dim=32,
        num_layers=2,
        num_heads=4,
        timesteps=20,
        backbone_type="mlp",
        fusion_type="concat",
        lr=1e-3,
    )
    model = ARLDMTransformer(cfg)
    optimizer, _ = model.configure_optimizers()
    batch = _build_batch()
    losses = model.training_step(batch)
    losses["loss"].backward()
    optimizer.step()
    assert losses["loss"].dim() == 0
    assert torch.isfinite(losses["loss"])
    assert_close(losses["loss"], losses["diffusion_loss"])


def test_ar_ldm_sample_latent_shape_and_finiteness() -> None:
    """Tests reverse diffusion sampling shape and finiteness."""
    cfg = ARLDMConfig(
        latent_dim=6,
        cond_motion_dim=12,
        text_dim=16,
        model_dim=32,
        num_layers=1,
        num_heads=4,
        timesteps=10,
        backbone_type="transformer",
        fusion_type="cross_attention",
    )
    model = ARLDMTransformer(cfg)
    cond_motion = torch.randn(3, 12)
    cond_text = torch.randn(3, 16)
    samples = model.sample_latent(cond_motion=cond_motion, cond_text=cond_text, steps=5)
    assert samples.shape == (3, 6)
    assert_close(samples.isfinite().float().mean(), torch.tensor(1.0))

