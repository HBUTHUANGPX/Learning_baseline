"""Minimal unit tests for ConvVAE, VQ-VAE, and FSQ-VAE."""

from __future__ import annotations

import torch

from modules.vae import ConvVAE
from modules.vqvae import FSQVAE, VQVAE


def test_conv_vae_forward_and_loss() -> None:
    """Tests ConvVAE forward shape and scalar loss outputs."""
    model = ConvVAE(latent_dim=8)
    x = torch.rand(4, 1, 28, 28)
    outputs = model(x)
    losses = model.loss_function(x, outputs)

    assert outputs["x_hat"].shape == (4, 1, 28, 28)
    assert outputs["mu"].shape == (4, 8)
    assert outputs["logvar"].shape == (4, 8)
    assert set(losses.keys()) == {"loss", "recon_loss", "kl_loss"}


def test_vqvae_forward_and_loss() -> None:
    """Tests VQ-VAE forward path and quantization loss outputs."""
    model = VQVAE(embedding_dim=8, num_embeddings=32, beta=0.25)
    x = torch.rand(3, 1, 28, 28)
    outputs = model(x)
    losses = model.loss_function(x, outputs)

    assert outputs["x_hat"].shape == (3, 1, 28, 28)
    assert outputs["z_e"].shape[1] == 8
    assert set(losses.keys()) == {"loss", "recon_loss", "quant_loss", "perplexity"}
    assert losses["loss"].dim() == 0


def test_fsqvae_forward_and_loss() -> None:
    """Tests FSQ-VAE forward path and scalar loss outputs."""
    model = FSQVAE(embedding_dim=8, fsq_levels=6, beta=0.25)
    x = torch.rand(2, 1, 28, 28)
    outputs = model(x)
    losses = model.loss_function(x, outputs)

    assert outputs["x_hat"].shape == (2, 1, 28, 28)
    assert set(losses.keys()) == {"loss", "recon_loss", "quant_loss", "perplexity"}
    assert losses["loss"].dim() == 0
