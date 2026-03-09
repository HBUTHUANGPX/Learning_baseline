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

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.latent_dim = latent_dim
        self.step = 0

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
    ldm = _DummyLDM(latent_dim=4)
    dec = _DummyDecoder(frame_dim=d, frames_per_step=m)
    gen = ARLatentRolloutGenerator(ldm_model=ldm, fsq_decoder=dec, config=cfg)

    seed = torch.zeros(n + 1, d)
    text = torch.zeros(8)
    seq, future = gen.rollout(seed_window=seed, text_token=text, num_steps=2)
    assert seq.shape == (n + 1 + 2 * m, d)
    assert future.shape == (2 * m, d)
    assert_close(seq.isfinite().float().mean(), torch.tensor(1.0))


def test_ar_rollout_sliding_history_uses_last_n_generated_frames() -> None:
    """Tests next-step condition motion uses last n frames from generated future."""
    n, m, d = 2, 3, 1
    cfg = RolloutConfig(history_frames=n, future_frames=m, frame_dim=d, diffusion_steps=5)
    ldm = _DummyLDM(latent_dim=4)
    dec = _DummyDecoder(frame_dim=d, frames_per_step=m)
    gen = ARLatentRolloutGenerator(ldm_model=ldm, fsq_decoder=dec, config=cfg)

    seed = torch.zeros(n + 1, d)
    text = torch.zeros(8)
    _ = gen.rollout(seed_window=seed, text_token=text, num_steps=2)

    first_cond = dec.cond_inputs[0].reshape(n + 1, d)
    second_cond = dec.cond_inputs[1].reshape(n + 1, d)
    assert_close(first_cond, torch.zeros_like(first_cond))
    # Step-1 generated block is [1, 2, 3], so next condition is [2, 3, 3].
    expected_second = torch.tensor([[2.0], [3.0], [3.0]])
    assert_close(second_cond, expected_second)


def test_ar_rollout_invalid_seed_shape_raises() -> None:
    """Tests rollout validates seed window shape."""
    cfg = RolloutConfig(history_frames=2, future_frames=3, frame_dim=1, diffusion_steps=5)
    ldm = _DummyLDM(latent_dim=4)
    dec = _DummyDecoder(frame_dim=1, frames_per_step=3)
    gen = ARLatentRolloutGenerator(ldm_model=ldm, fsq_decoder=dec, config=cfg)
    with pytest.raises(ValueError):
        _ = gen.rollout(seed_window=torch.zeros(2, 1), text_token=torch.zeros(8), num_steps=1)

