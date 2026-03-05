"""VQ-VAE and FSQ-VAE model definitions."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import VQVAEConfig
from .conv import VQConvDecoder, VQConvEncoder, flatten_spatial, unflatten_spatial
from .quantizers import FSQQuantizer, VectorQuantizer


class VQVAE(nn.Module):
    """Vector-Quantized VAE for 28x28 grayscale images."""

    def __init__(
        self,
        embedding_dim: int = 16,
        num_embeddings: int = 128,
        beta: float = 0.25,
        recon_loss_mode: str = "auto",
        config: VQVAEConfig | None = None,
    ) -> None:
        """Initializes a VQ-VAE model.

        Args:
            embedding_dim: Channel size of latent feature maps.
            num_embeddings: Number of codebook vectors.
            beta: Commitment coefficient for quantization.
            recon_loss_mode: Reconstruction loss mode in ``{"auto","bce","mse"}``.
            config: Optional VQVAEConfig. If provided, it overrides architecture
                and quantization arguments.
        """
        super().__init__()
        if config is None:
            config = VQVAEConfig(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                beta=beta,
                recon_loss_mode=recon_loss_mode,
            )
        self.config = config
        self.embedding_dim = int(config.embedding_dim)
        self.num_embeddings = int(config.num_embeddings)
        self.beta = float(config.beta)
        self.recon_loss_mode = str(config.recon_loss_mode)

        self.encoder = self._build_encoder(self.embedding_dim)
        self.decoder = self._build_decoder(self.embedding_dim)
        self.quantizer = self._build_quantizer(
            embedding_dim=self.embedding_dim,
            num_embeddings=self.num_embeddings,
            beta=self.beta,
        )

    def _build_encoder(self, embedding_dim: int) -> nn.Module:
        """Builds encoder module.

        Args:
            embedding_dim: Channel size of latent feature map.

        Returns:
            Encoder module instance.
        """
        return VQConvEncoder(
            embedding_dim=embedding_dim,
            image_shape=(
                self.config.image.channels,
                self.config.image.height,
                self.config.image.width,
            ),
            conv_channels=self.config.encoder_channels,
        )

    def _build_decoder(self, embedding_dim: int) -> nn.Module:
        """Builds decoder module.

        Args:
            embedding_dim: Channel size of latent feature map.

        Returns:
            Decoder module instance.
        """
        return VQConvDecoder(
            embedding_dim=embedding_dim,
            output_channels=self.config.image.channels,
            decoder_channels=self.config.decoder_channels,
        )

    def _build_quantizer(
        self, embedding_dim: int, num_embeddings: int, beta: float
    ) -> nn.Module:
        """Builds quantizer module.

        Args:
            embedding_dim: Latent embedding dimension.
            num_embeddings: Number of codebook entries.
            beta: Commitment coefficient.

        Returns:
            Quantizer module instance.
        """
        return VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, beta=beta
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs forward pass for VQ-VAE.

        Args:
            x: Input tensor with shape ``[B, 1, 28, 28]``.

        Returns:
            Dictionary containing reconstruction and quantization stats.
        """
        z_e_map = self.encoder(x)
        z_e_flat, meta = flatten_spatial(z_e_map)
        q = self.quantizer(z_e_flat)
        z_q_map = unflatten_spatial(q["z_q"], meta)
        x_hat = self.decoder(z_q_map)
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
        """Computes VQ-VAE loss.

        Args:
            x: Ground-truth input image batch.
            outputs: Forward outputs from ``self.forward``.

        Returns:
            Dictionary with scalar loss terms.
        """
        recon = self._reconstruction_loss(
            outputs["x_hat"], x, mode=self.recon_loss_mode
        )
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
        """Computes VQ reconstruction loss with auto BCE/MSE fallback.

        Args:
            x_hat: Reconstruction tensor.
            x: Ground-truth tensor.
            mode: One of ``{"auto","bce","mse"}``.

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


class FSQVAE(VQVAE):
    """FSQ-VAE using finite scalar quantization instead of vector codebook."""

    def __init__(
        self,
        embedding_dim: int = 16,
        fsq_levels: int = 8,
        beta: float = 0.25,
        recon_loss_mode: str = "auto",
        config: VQVAEConfig | None = None,
    ) -> None:
        """Initializes FSQ-VAE.

        Args:
            embedding_dim: Channel size of latent feature maps.
            fsq_levels: Number of scalar bins in FSQ.
            beta: Commitment coefficient for FSQ.
            recon_loss_mode: Reconstruction loss mode in ``{"auto","bce","mse"}``.
            config: Optional VQVAEConfig. If provided, it overrides architecture
                and FSQ-specific arguments.
        """
        if config is None:
            config = VQVAEConfig(
                embedding_dim=embedding_dim,
                beta=beta,
                fsq_levels=fsq_levels,
                recon_loss_mode=recon_loss_mode,
            )
        self.fsq_levels = int(config.fsq_levels)
        super().__init__(
            embedding_dim=config.embedding_dim,
            num_embeddings=self.fsq_levels,
            beta=config.beta,
            recon_loss_mode=config.recon_loss_mode,
            config=config,
        )

    def _build_quantizer(
        self, embedding_dim: int, num_embeddings: int, beta: float
    ) -> nn.Module:
        """Builds FSQ quantizer while keeping parent initialization chain.

        Args:
            embedding_dim: Latent embedding dimension. Unused for FSQ bins.
            num_embeddings: Placeholder to keep parent hook signature stable.
            beta: Commitment coefficient.

        Returns:
            FSQ quantizer instance.
        """
        del embedding_dim, num_embeddings
        return FSQQuantizer(levels=self.fsq_levels, beta=beta)
