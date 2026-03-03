"""Encoder implementations for VAE-family models."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

from .mlp import MLP


class MLPGaussianEncoder(nn.Module):
    """Gaussian encoder that outputs latent mean and log-variance.

    The encoder maps flattened input vectors into latent parameters used by the
    reparameterization trick in VAE variants.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        latent_dim: int,
        activation: str = "relu",
    ) -> None:
        """Initializes the encoder.

        Args:
            input_dim: Flattened input feature size.
            hidden_dims: Hidden MLP layer sizes.
            latent_dim: Latent space dimension.
            activation: Hidden activation name.
        """
        super().__init__()
        hidden_list = list(hidden_dims)
        self.backbone = MLP(
            in_dim=input_dim,
            hidden_dims=hidden_list,
            out_dim=hidden_list[-1] if hidden_list else input_dim,
            activation=activation,
            final_activation=activation,
        )
        feat_dim = hidden_list[-1] if hidden_list else input_dim
        self.mu_head = nn.Linear(feat_dim, latent_dim)
        self.logvar_head = nn.Linear(feat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes posterior parameters.

        Args:
            x: Input tensor with shape ``[batch, input_dim]``.

        Returns:
            A tuple ``(mu, logvar)`` each with shape ``[batch, latent_dim]``.
        """
        h = self.backbone(x)
        return self.mu_head(h), self.logvar_head(h)
