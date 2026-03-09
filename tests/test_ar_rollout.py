"""Tests for autoregressive latent rollout module.

Coverage target: > 85% for this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch.testing import assert_close

from modules.ar_rollout import ARLatentRolloutGenerator, RolloutConfig


class _DummyLDM(torch.nn.Module):
    """Dummy latent sampler with deterministic per-step latent values."""

    def __init__(self, latent_dim: int, cond_motion_dim: int) -> None:
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.latent_dim = latent_dim
        self.step = 0
        self.config = type("Cfg", (), {"cond_motion_dim": cond_motion_dim})()

    def sample_latent(
        self,
        cond_motion: torch.Tensor,
        cond_text: torch.Tensor,
        steps: int | None = None,
    ) -> torch.Tensor:
        self.step += 1
        value = float(self.step)
        return torch.full((cond_motion.shape[0], self.latent_dim), value, device=cond_motion.device)


class _DummyDecoder:
    """Dummy decoder that records conditions and emits deterministic frame blocks."""

    def __init__(self, frame_dim: int, frames_per_step: int) -> None:
        self.frame_dim = frame_dim
        self.frames_per_step = frames_per_step
        self.cond_inputs: list[torch.Tensor] = []

    def decode(self, latent: torch.Tensor, cond_motion: torch.Tensor) -> torch.Tensor:
        self.cond_inputs.append(cond_motion.detach().cpu())
        base = latent[:, :1]
        frames = []
        for idx in range(self.frames_per_step):
            frames.append(base + float(idx))
        stacked = torch.cat(frames, dim=1)  # [B, frames_per_step]
        if self.frame_dim > 1:
            stacked = stacked.unsqueeze(-1).repeat(1, 1, self.frame_dim).reshape(latent.shape[0], -1)
        return stacked


def test_ar_rollout_shapes() -> None:
    """Tests rollout output shape for minimal deterministic setup."""
    n, m, d = 2, 3, 1
    cfg = RolloutConfig(history_frames=n, future_frames=m, frame_dim=d, diffusion_steps=5)
    ldm = _DummyLDM(latent_dim=4, cond_motion_dim=n * d)
    dec = _DummyDecoder(frame_dim=d, frames_per_step=1 + m)
    gen = ARLatentRolloutGenerator(ldm_model=ldm, fsq_decoder=dec, config=cfg)
    dec.decoder_condition_dim = n * d
    dec.target_dim = (1 + m) * d

    seed = torch.zeros(n, d)
    text = torch.zeros(8)
    seq, future = gen.rollout(seed_history=seed, text_token=text, num_steps=2)
    assert seq.shape == (n + 2 * (1 + m), d)
    assert future.shape == (2 * (1 + m), d)
    assert_close(seq.isfinite().float().mean(), torch.tensor(1.0))


def test_ar_rollout_sliding_history_uses_last_n_generated_frames() -> None:
    """Tests next-step condition motion uses last n frames from generated future."""
    n, m, d = 2, 3, 1
    cfg = RolloutConfig(history_frames=n, future_frames=m, frame_dim=d, diffusion_steps=5)
    ldm = _DummyLDM(latent_dim=4, cond_motion_dim=n * d)
    dec = _DummyDecoder(frame_dim=d, frames_per_step=1 + m)
    gen = ARLatentRolloutGenerator(ldm_model=ldm, fsq_decoder=dec, config=cfg)
    dec.decoder_condition_dim = n * d
    dec.target_dim = (1 + m) * d

    seed = torch.zeros(n, d)
    text = torch.zeros(8)
    _ = gen.rollout(seed_history=seed, text_token=text, num_steps=2)

    first_cond = dec.cond_inputs[0].reshape(n, d)
    second_cond = dec.cond_inputs[1].reshape(n, d)
    assert_close(first_cond, torch.zeros_like(first_cond))
    # Step-1 generated block is [1, 2, 3, 4], so next history is [3, 4].
    expected_second = torch.tensor([[3.0], [4.0]])
    assert_close(second_cond, expected_second)


def test_ar_rollout_invalid_seed_shape_raises() -> None:
    """Tests rollout validates seed window shape."""
    cfg = RolloutConfig(history_frames=2, future_frames=3, frame_dim=1, diffusion_steps=5)
    ldm = _DummyLDM(latent_dim=4, cond_motion_dim=2)
    dec = _DummyDecoder(frame_dim=1, frames_per_step=4)
    gen = ARLatentRolloutGenerator(ldm_model=ldm, fsq_decoder=dec, config=cfg)
    dec.decoder_condition_dim = 2
    dec.target_dim = 4
    with pytest.raises(ValueError):
        _ = gen.rollout(seed_history=torch.zeros(3, 1), text_token=torch.zeros(8), num_steps=1)


def test_ar_rollout_decoder_condition_dim_mismatch_raises() -> None:
    """Tests rollout validates decoder condition dimension against config."""
    cfg = RolloutConfig(history_frames=2, future_frames=3, frame_dim=1, diffusion_steps=5)
    ldm = _DummyLDM(latent_dim=4, cond_motion_dim=2)
    dec = _DummyDecoder(frame_dim=1, frames_per_step=4)
    dec.decoder_condition_dim = 3
    dec.target_dim = 4
    gen = ARLatentRolloutGenerator(ldm_model=ldm, fsq_decoder=dec, config=cfg)
    with pytest.raises(ValueError, match="condition dim mismatch"):
        _ = gen.rollout(seed_history=torch.zeros(2, 1), text_token=torch.zeros(8), num_steps=1)


def test_ar_rollout_invalid_n_gt_1_plus_m_raises() -> None:
    """Tests rollout rejects invalid temporal relation n > 1 + m."""
    cfg = RolloutConfig(history_frames=5, future_frames=3, frame_dim=1, diffusion_steps=5)
    ldm = _DummyLDM(latent_dim=4, cond_motion_dim=5)
    dec = _DummyDecoder(frame_dim=1, frames_per_step=4)
    dec.decoder_condition_dim = 5
    dec.target_dim = 4
    gen = ARLatentRolloutGenerator(ldm_model=ldm, fsq_decoder=dec, config=cfg)
    with pytest.raises(ValueError, match="n <= 1 \\+ m"):
        _ = gen.rollout(seed_history=torch.zeros(5, 1), text_token=torch.zeros(8), num_steps=1)
