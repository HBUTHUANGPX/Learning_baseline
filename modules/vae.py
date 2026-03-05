"""VAE-family model definitions."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import ConvDecoder, ConvGaussianEncoder
from .configs import ConvVAEConfig, MLPVAEConfig
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

    def forward(
        self, x: torch.Tensor, sample: bool | None = None
    ) -> Dict[str, torch.Tensor]:
        """Runs the full VAE forward process.

        Args:
            x: Input tensor with shape ``[batch, input_dim]``.
            sample: Whether to sample latent vectors with reparameterization.
                If ``None``, sampling follows module mode (train: sample,
                eval: deterministic ``z=mu``).

        Returns:
            A dictionary containing reconstruction and latent statistics.
        """
        mu, logvar = self.encode(x)
        should_sample = self.training if sample is None else sample
        z = self.reparameterize(mu, logvar) if should_sample else mu
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "mu": mu, "logvar": logvar, "z": z}

    @staticmethod
    def reconstruction_loss(
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mode: str = "auto",
    ) -> torch.Tensor:
        """Computes reconstruction loss with automatic mode selection.

        Args:
            x_hat: Reconstruction tensor.
            x: Ground-truth tensor.
            mode: One of ``"auto"``, ``"bce"``, or ``"mse"``.

        Returns:
            Batch-averaged reconstruction loss.
        """
        mode = mode.lower().strip()
        if mode not in {"auto", "bce", "mse"}:
            raise ValueError(f"Unsupported reconstruction loss mode: {mode}")

        if mode == "auto":
            x_in_range = bool(torch.logical_and(x >= 0.0, x <= 1.0).all().item())
            xhat_in_range = bool(
                torch.logical_and(x_hat >= 0.0, x_hat <= 1.0).all().item()
            )
            mode = "bce" if (x_in_range and xhat_in_range) else "mse"

        if mode == "bce":
            return F.binary_cross_entropy(x_hat, x, reduction="sum") / x.shape[0]
        return F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]


class VanillaVAE(BaseVAE):
    """Standard VAE with MLP encoder/decoder and adaptive reconstruction loss."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Iterable[int] = (512, 256),
        activation: str = "relu",
        recon_loss_mode: str = "auto",
        config: MLPVAEConfig | None = None,
    ) -> None:
        """Initializes a Vanilla VAE.

        Args:
            input_dim: Flattened input dimension.
            latent_dim: Latent space dimension.
            hidden_dims: Shared hidden dimensions for encoder and decoder.
            activation: Activation name for hidden layers.
            recon_loss_mode: Reconstruction loss mode in ``{"auto","bce","mse"}``.
            config: Optional MLPVAEConfig. If provided, it overrides architecture
                arguments above.
        """
        super().__init__()
        if config is not None:
            input_dim = config.input_dim
            latent_dim = config.latent_dim
            hidden_dims = config.hidden_dims
            activation = config.activation
            recon_loss_mode = config.recon_loss_mode
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
        self.recon_loss_mode = recon_loss_mode

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes input into Gaussian posterior parameters."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vectors into reconstructed vectors."""
        return self.decoder(z)

    def loss_function(
        self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Computes Vanilla VAE objective.

        Args:
            x: Ground-truth batch with shape ``[batch, input_dim]``.
            outputs: Forward outputs from ``self.forward``.

        Returns:
            Dictionary with ``loss``, ``recon_loss``, and ``kl_loss``.
        """
        x_hat, mu, logvar = outputs["x_hat"], outputs["mu"], outputs["logvar"]
        recon = self.reconstruction_loss(x_hat, x, mode=self.recon_loss_mode)
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
        recon_loss_mode: str = "auto",
        config: MLPVAEConfig | None = None,
    ) -> None:
        """Initializes a Beta-VAE model.

        Args:
            input_dim: Flattened input dimension.
            latent_dim: Latent space dimension.
            beta: KL divergence scaling factor.
            hidden_dims: Hidden dimensions for MLP blocks.
            activation: Hidden activation name.
            recon_loss_mode: Reconstruction loss mode in ``{"auto","bce","mse"}``.
            config: Optional MLPVAEConfig. If provided, it overrides architecture
                arguments. ``beta`` still uses explicit argument.
        """
        if config is not None:
            input_dim = config.input_dim
            latent_dim = config.latent_dim
            hidden_dims = config.hidden_dims
            activation = config.activation
            beta = config.beta
            recon_loss_mode = config.recon_loss_mode
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            recon_loss_mode=recon_loss_mode,
        )
        self.beta = float(beta)

    def loss_function(
        self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Computes Beta-VAE objective.

        Args:
            x: Ground-truth batch.
            outputs: Forward outputs from ``self.forward``.

        Returns:
            Dictionary with scalar training losses.
        """
        base = super().loss_function(x, outputs)
        total = base["recon_loss"] + self.beta * base["kl_loss"]
        return {
            "loss": total,
            "recon_loss": base["recon_loss"],
            "kl_loss": base["kl_loss"],
        }


class ConvVAE(BaseVAE):
    """Convolutional VAE variant for ``[B, 1, 28, 28]`` image inputs."""

    def __init__(
        self,
        latent_dim: int = 16,
        recon_loss_mode: str = "auto",
        config: ConvVAEConfig | None = None,
    ) -> None:
        """Initializes a convolutional VAE.

        Args:
            latent_dim: Latent space dimension.
            recon_loss_mode: Reconstruction loss mode in ``{"auto","bce","mse"}``.
            config: Optional ConvVAEConfig that controls image and structure
                parameters.
        """
        super().__init__()
        if config is None:
            config = ConvVAEConfig(
                latent_dim=latent_dim, recon_loss_mode=recon_loss_mode
            )
        self.config = config
        self.encoder = ConvGaussianEncoder(
            latent_dim=config.latent_dim,
            image_shape=(
                config.image.channels,
                config.image.height,
                config.image.width,
            ),
            conv_channels=config.encoder_channels,
            bottleneck_dim=config.bottleneck_dim,
        )
        self.decoder = ConvDecoder(
            latent_dim=config.latent_dim,
            feature_shape=self.encoder.feature_shape,
            output_channels=config.image.channels,
            decoder_channels=tuple(reversed(config.encoder_channels[:-1])),
        )
        self.recon_loss_mode = config.recon_loss_mode

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes image input into Gaussian parameters."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vectors back to images."""
        return self.decoder(z)

    def loss_function(
        self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Computes convolutional VAE objective.

        Args:
            x: Ground-truth image tensor ``[B, 1, 28, 28]``.
            outputs: Forward outputs from ``self.forward``.

        Returns:
            Dictionary with scalar loss terms.
        """
        x_hat, mu, logvar = outputs["x_hat"], outputs["mu"], outputs["logvar"]
        recon = self.reconstruction_loss(x_hat, x, mode=self.recon_loss_mode)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        total = recon + kl
        return {"loss": total, "recon_loss": recon, "kl_loss": kl}
