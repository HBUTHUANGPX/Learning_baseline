"""Minimal tests for frame-level VQ-VAE and FSQ-VAE models."""

from __future__ import annotations

import torch

from modules.vqvae import FrameFSQVAE, FrameVQVAE


def test_frame_vqvae_forward_and_loss() -> None:
    """Tests FrameVQVAE forward outputs and scalar losses."""
    model = FrameVQVAE(input_dim=20, embedding_dim=8, hidden_dim=32, num_embeddings=16)
    x = torch.rand(4, 20)
    outputs = model(x)
    losses = model.loss_function(x, outputs)

    assert outputs["x_hat"].shape == (4, 20)
    assert set(losses.keys()) == {"loss", "recon_loss", "quant_loss", "perplexity"}
    assert losses["loss"].dim() == 0


def test_frame_fsqvae_forward_and_loss() -> None:
    """Tests FrameFSQVAE forward outputs and scalar losses."""
    model = FrameFSQVAE(input_dim=20, embedding_dim=8, hidden_dim=32, fsq_levels=6)
    x = torch.rand(3, 20)
    outputs = model(x)
    losses = model.loss_function(x, outputs)

    assert outputs["x_hat"].shape == (3, 20)
    assert set(losses.keys()) == {"loss", "recon_loss", "quant_loss", "perplexity"}
    assert losses["loss"].dim() == 0
