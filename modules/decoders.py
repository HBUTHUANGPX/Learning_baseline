"""Decoder implementations for VAE-family models."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .mlp import MLP


class MLPDecoder(nn.Module):
    """MLP decoder that maps latent vectors back to data space."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        activation: str = "relu",
    ) -> None:
        """Initializes the decoder.

        Args:
            latent_dim: Latent vector dimension.
            hidden_dims: Hidden MLP layer sizes.
            output_dim: Reconstruction output dimension.
            activation: Hidden activation name.
        """
        super().__init__()
        self.net = MLP(
            in_dim=latent_dim,
            hidden_dims=hidden_dims,
            out_dim=output_dim,
            activation=activation,
            final_activation="sigmoid",
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vectors into reconstruction probabilities.

        Args:
            z: Latent tensor with shape ``[batch, latent_dim]``.

        Returns:
            Reconstructed tensor with shape ``[batch, output_dim]``.
        """
        return self.net(z)
