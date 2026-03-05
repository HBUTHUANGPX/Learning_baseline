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


def test_vanilla_vae_auto_recon_loss_handles_continuous_targets() -> None:
    """Tests auto recon loss falls back from BCE for non-[0,1] continuous inputs."""
    model = VanillaVAE(input_dim=16, latent_dim=4, hidden_dims=(8, 4))
    # Motion-like continuous values that violate BCE target range.
    x = torch.randn(7, 16) * 3.0
    outputs = model(x)
    losses = model.loss_function(x, outputs)
    assert torch.isfinite(losses["recon_loss"]).item()
    assert torch.isfinite(losses["loss"]).item()
