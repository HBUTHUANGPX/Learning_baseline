"""Minimal tests for context-conditioned VQ-VAE and FSQ-VAE models."""

from __future__ import annotations

import torch

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
    assert set(losses.keys()) == {"loss", "recon_loss", "quant_loss", "perplexity"}
    assert losses["loss"].dim() == 0
