"""Frame-level convolutional models for vector-shaped observations."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import FSQQuantizer, VectorQuantizer
from .vae import BaseVAE


class FrameConvVAE(BaseVAE):
    """Conv1d-based VAE for frame vectors ``[B, D]``."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_channels: int = 64,
        bottleneck_dim: int = 256,
        recon_loss_mode: str = "auto",
    ) -> None:
        """Initializes frame-level ConvVAE.

        Args:
            input_dim: Feature dimension per frame.
            latent_dim: Latent dimension of Gaussian posterior.
            hidden_channels: Hidden channel count for Conv1d blocks.
            bottleneck_dim: Fully-connected bottleneck width.
            recon_loss_mode: Reconstruction loss mode in ``{"auto","bce","mse"}``.
        """
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_channels = int(hidden_channels)
        self.bottleneck_dim = int(bottleneck_dim)
        self.recon_loss_mode = str(recon_loss_mode)

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_flat_dim = self.hidden_channels * self.input_dim
        self.encoder_fc = nn.Linear(self.encoder_flat_dim, self.bottleneck_dim)
        self.mu = nn.Linear(self.bottleneck_dim, self.latent_dim)
        self.logvar = nn.Linear(self.bottleneck_dim, self.latent_dim)

        self.decoder_fc = nn.Linear(self.latent_dim, self.hidden_channels * self.input_dim)
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_channels, 1, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes frame vectors into posterior parameters.

        Args:
            x: Input frame tensor with shape ``[B, D]``.

        Returns:
            Tuple ``(mu, logvar)`` with shape ``[B, latent_dim]``.
        """
        x_c = x.unsqueeze(1)
        h = self.encoder_conv(x_c)
        h = h.reshape(x.shape[0], -1)
        h = torch.relu(self.encoder_fc(h))
        return self.mu(h), self.logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vectors back to frame vectors.

        Args:
            z: Latent tensor with shape ``[B, latent_dim]``.

        Returns:
            Reconstruction tensor with shape ``[B, D]``.
        """
        h = torch.relu(self.decoder_fc(z))
        h = h.view(z.shape[0], self.hidden_channels, self.input_dim)
        return self.decoder_conv(h).squeeze(1)

    def loss_function(
        self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Computes frame-level ConvVAE objective."""
        x_hat, mu, logvar = outputs["x_hat"], outputs["mu"], outputs["logvar"]
        recon = self.reconstruction_loss(x_hat, x, mode=self.recon_loss_mode)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        total = recon + kl
        return {"loss": total, "recon_loss": recon, "kl_loss": kl}


class FrameVQVAE(nn.Module):
    """Conv1d VQ-VAE for frame vectors ``[B, D]``."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 16,
        hidden_channels: int = 64,
        num_embeddings: int = 128,
        beta: float = 0.25,
        recon_loss_mode: str = "auto",
    ) -> None:
        """Initializes frame-level VQ-VAE.

        Args:
            input_dim: Feature dimension per frame.
            embedding_dim: Latent channel count before quantization.
            hidden_channels: Hidden channel count for Conv1d blocks.
            num_embeddings: Number of VQ codebook vectors.
            beta: Commitment loss coefficient.
            recon_loss_mode: Reconstruction loss mode in ``{"auto","bce","mse"}``.
        """
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.hidden_channels = int(hidden_channels)
        self.recon_loss_mode = str(recon_loss_mode)

        self.encoder = nn.Sequential(
            nn.Conv1d(1, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_channels, self.embedding_dim, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_channels, 1, kernel_size=3, padding=1),
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=int(num_embeddings),
            embedding_dim=self.embedding_dim,
            beta=float(beta),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs frame-level VQ forward pass."""
        x_c = x.unsqueeze(1)
        z_e_map = self.encoder(x_c)
        z_e_flat = z_e_map.transpose(1, 2).reshape(-1, self.embedding_dim)
        q = self.quantizer(z_e_flat)
        z_q_map = q["z_q"].view(x.shape[0], self.input_dim, self.embedding_dim).transpose(1, 2)
        x_hat = self.decoder(z_q_map).squeeze(1)
        return {
            "x_hat": x_hat,
            "z_e": z_e_map,
            "z_q": z_q_map,
            "indices": q["indices"],
            "quant_loss": q["quant_loss"],
            "perplexity": q["perplexity"],
        }

    def loss_function(
        self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Computes frame-level VQ objective."""
        recon = self._reconstruction_loss(outputs["x_hat"], x, mode=self.recon_loss_mode)
        quant = outputs["quant_loss"]
        total = recon + quant
        return {
            "loss": total,
            "recon_loss": recon,
            "quant_loss": quant,
            "perplexity": outputs["perplexity"],
        }

    @staticmethod
    def _reconstruction_loss(
        x_hat: torch.Tensor, x: torch.Tensor, mode: str = "auto"
    ) -> torch.Tensor:
        """Computes reconstruction loss for frame VQ models."""
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


class FrameFSQVAE(FrameVQVAE):
    """Conv1d FSQ-VAE for frame vectors ``[B, D]``."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 16,
        hidden_channels: int = 64,
        fsq_levels: int = 8,
        beta: float = 0.25,
        recon_loss_mode: str = "auto",
    ) -> None:
        """Initializes frame-level FSQ-VAE.

        Args:
            input_dim: Feature dimension per frame.
            embedding_dim: Latent channel count before quantization.
            hidden_channels: Hidden channel count for Conv1d blocks.
            fsq_levels: Number of scalar quantization levels.
            beta: Commitment loss coefficient.
            recon_loss_mode: Reconstruction loss mode in ``{"auto","bce","mse"}``.
        """
        super().__init__(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_channels=hidden_channels,
            num_embeddings=fsq_levels,
            beta=beta,
            recon_loss_mode=recon_loss_mode,
        )
        self.quantizer = FSQQuantizer(levels=int(fsq_levels), beta=float(beta))
