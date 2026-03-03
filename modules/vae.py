"""VAE-family model definitions."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import ConvDecoder, ConvGaussianEncoder
from .decoders import MLPDecoder
from .encoders import MLPGaussianEncoder


class BaseVAE(nn.Module):
    """Abstract base class for VAE-style models."""

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes inputs into posterior Gaussian parameters."""
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent codes into reconstructed inputs."""
        raise NotImplementedError

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Applies the reparameterization trick.

        Args:
            mu: Posterior mean.
            logvar: Posterior log variance.

        Returns:
            Sampled latent tensor with the same shape as ``mu``.
        """
        # Sampling noise in latent space enables stochastic gradients.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs the full VAE forward process.

        Args:
            x: Input tensor with shape ``[batch, input_dim]``.

        Returns:
            A dictionary containing reconstruction and latent statistics.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "mu": mu, "logvar": logvar, "z": z}


class VanillaVAE(BaseVAE):
    """Standard VAE with MLP encoder/decoder and BCE reconstruction loss."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Iterable[int] = (512, 256),
        activation: str = "relu",
    ) -> None:
        """Initializes a Vanilla VAE.

        Args:
            input_dim: Flattened input dimension.
            latent_dim: Latent space dimension.
            hidden_dims: Shared hidden dimensions for encoder and decoder.
            activation: Activation name for hidden layers.
        """
        super().__init__()
        hidden_list = list(hidden_dims)
        self.encoder = MLPGaussianEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_list,
            latent_dim=latent_dim,
            activation=activation,
        )
        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            hidden_dims=list(reversed(hidden_list)),
            output_dim=input_dim,
            activation=activation,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes input into Gaussian posterior parameters."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vectors into reconstructed vectors."""
        return self.decoder(z)

    def loss_function(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Computes Vanilla VAE objective.

        Args:
            x: Ground-truth batch with shape ``[batch, input_dim]``.
            outputs: Forward outputs from ``self.forward``.

        Returns:
            Dictionary with ``loss``, ``recon_loss``, and ``kl_loss``.
        """
        x_hat, mu, logvar = outputs["x_hat"], outputs["mu"], outputs["logvar"]
        recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.shape[0]
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        total = recon + kl
        return {"loss": total, "recon_loss": recon, "kl_loss": kl}


class BetaVAE(VanillaVAE):
    """Beta-VAE that scales KL divergence with configurable beta."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        beta: float = 4.0,
        hidden_dims: Iterable[int] = (512, 256),
        activation: str = "relu",
    ) -> None:
        """Initializes a Beta-VAE model.

        Args:
            input_dim: Flattened input dimension.
            latent_dim: Latent space dimension.
            beta: KL divergence scaling factor.
            hidden_dims: Hidden dimensions for MLP blocks.
            activation: Hidden activation name.
        """
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        self.beta = float(beta)

    def loss_function(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Computes Beta-VAE objective.

        Args:
            x: Ground-truth batch.
            outputs: Forward outputs from ``self.forward``.

        Returns:
            Dictionary with scalar training losses.
        """
        base = super().loss_function(x, outputs)
        total = base["recon_loss"] + self.beta * base["kl_loss"]
        return {"loss": total, "recon_loss": base["recon_loss"], "kl_loss": base["kl_loss"]}


class ConvVAE(BaseVAE):
    """Convolutional VAE variant for ``[B, 1, 28, 28]`` image inputs."""

    def __init__(self, latent_dim: int = 16) -> None:
        """Initializes a convolutional VAE.

        Args:
            latent_dim: Latent space dimension.
        """
        super().__init__()
        self.encoder = ConvGaussianEncoder(latent_dim=latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes image input into Gaussian parameters."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vectors back to images."""
        return self.decoder(z)

    def loss_function(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Computes convolutional VAE objective.

        Args:
            x: Ground-truth image tensor ``[B, 1, 28, 28]``.
            outputs: Forward outputs from ``self.forward``.

        Returns:
            Dictionary with scalar loss terms.
        """
        x_hat, mu, logvar = outputs["x_hat"], outputs["mu"], outputs["logvar"]
        recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.shape[0]
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        total = recon + kl
        return {"loss": total, "recon_loss": recon, "kl_loss": kl}
