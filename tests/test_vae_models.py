"""Minimal unit tests for VAE model modules."""

from __future__ import annotations

import torch

from modules.vae import BetaVAE, VanillaVAE


def test_vanilla_vae_forward_shapes() -> None:
    """Tests whether VanillaVAE forward outputs expected tensor shapes."""
    model = VanillaVAE(input_dim=32, latent_dim=4, hidden_dims=(16, 8))
    x = torch.rand(5, 32)
    outputs = model(x)

    assert outputs["x_hat"].shape == (5, 32)
    assert outputs["mu"].shape == (5, 4)
    assert outputs["logvar"].shape == (5, 4)
    assert outputs["z"].shape == (5, 4)


def test_beta_vae_loss_contains_required_keys() -> None:
    """Tests whether BetaVAE loss function returns mandatory keys."""
    model = BetaVAE(input_dim=32, latent_dim=4, beta=2.0, hidden_dims=(16, 8))
    x = torch.rand(6, 32)
    outputs = model(x)
    losses = model.loss_function(x, outputs)

    assert set(losses.keys()) == {"loss", "recon_loss", "kl_loss"}
    assert losses["loss"].dim() == 0
