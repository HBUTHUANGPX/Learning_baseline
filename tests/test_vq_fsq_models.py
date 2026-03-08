"""Minimal tests for context-conditioned VQ-VAE and FSQ-VAE models."""

from __future__ import annotations

import torch

from modules.quantizers import FSQQuantizer
from modules.vqvae import FrameFSQVAE, FrameVQVAE


def test_frame_vqvae_forward_and_loss() -> None:
    """Tests FrameVQVAE forward outputs and scalar losses."""
    model = FrameVQVAE(
        encoder_input_dim=20,
        decoder_condition_dim=6,
        target_dim=12,
        embedding_dim=8,
        hidden_dim=32,
        num_embeddings=16,
    )
    enc = torch.rand(4, 20)
    cond = torch.rand(4, 6)
    target = torch.rand(4, 12)

    outputs = model(enc, cond)
    losses = model.loss_function(target, outputs)

    assert outputs["x_hat"].shape == (4, 12)
    assert set(losses.keys()) == {"loss", "recon_loss", "quant_loss", "perplexity"}
    assert losses["loss"].dim() == 0


def test_frame_fsqvae_forward_and_loss() -> None:
    """Tests FrameFSQVAE forward outputs and scalar losses."""
    model = FrameFSQVAE(
        encoder_input_dim=20,
        decoder_condition_dim=0,
        target_dim=10,
        embedding_dim=8,
        hidden_dim=32,
        fsq_levels=6,
    )
    enc = torch.rand(3, 20)
    cond = torch.rand(3, 0)
    target = torch.rand(3, 10)

    outputs = model(enc, cond)
    losses = model.loss_function(target, outputs)

    assert outputs["x_hat"].shape == (3, 10)
    assert set(losses.keys()) == {
        "loss",
        "recon_loss",
        "effective_bits",
        "avg_utilization",
        "level_histogram",
        "per_dim_usage",
    }
    assert losses["loss"].dim() == 0
    assert torch.allclose(losses["loss"], losses["recon_loss"])


def test_fsq_quantizer_returns_full_indices_and_usage_metrics() -> None:
    """Tests FSQ quantizer outputs full per-dimension indices and usage metrics."""
    quantizer = FSQQuantizer(levels=8)
    z_e = torch.randn(5, 7)
    out = quantizer(z_e)

    assert out["z_q"].shape == (5, 7)
    assert out["indices"].shape == (5, 7)
    assert out["indices"].dtype == torch.long
    assert "quant_loss" not in out
    assert out["level_histogram"].shape == (8,)
    assert out["per_dim_usage"].shape == (7, 8)
    assert out["avg_utilization"].dim() == 0
    assert out["effective_bits"].dim() == 0


def test_reconstruction_mode_rejects_auto() -> None:
    """Tests model rejects unsupported dynamic auto reconstruction mode."""
    try:
        _ = FrameVQVAE(
            encoder_input_dim=8,
            decoder_condition_dim=0,
            target_dim=4,
            recon_loss_mode="auto",
        )
    except ValueError:
        return
    raise AssertionError("FrameVQVAE must reject recon_loss_mode='auto'.")
