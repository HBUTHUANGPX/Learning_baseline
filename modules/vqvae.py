"""Frame-level VQ-VAE and FSQ-VAE models for motion reconstruction."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import FSQQuantizer, VectorQuantizer


class FrameVQVAE(nn.Module):
    """Vector-quantized autoencoder for frame vectors ``[B, D]``."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        num_embeddings: int = 512,
        beta: float = 0.25,
        recon_loss_mode: str = "mse",
    ) -> None:
        """Initializes a frame-level VQ-VAE model.

        Args:
            input_dim: Input frame feature dimension.
            embedding_dim: Latent embedding dimension before quantization.
            hidden_dim: Hidden MLP width.
            num_embeddings: Number of VQ codebook vectors.
            beta: Commitment loss weight.
            recon_loss_mode: Reconstruction loss mode, one of
                ``{"auto", "bce", "mse"}``.
        """
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.recon_loss_mode = str(recon_loss_mode)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.input_dim),
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=int(num_embeddings),
            embedding_dim=self.embedding_dim,
            beta=float(beta),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs one forward pass.

        Args:
            x: Input tensor with shape ``[B, D]``.

        Returns:
            Dictionary containing reconstruction and quantization outputs.
        """
        z_e = self.encoder(x)
        q = self.quantizer(z_e)
        x_hat = self.decoder(q["z_q"])
        return {
            "x_hat": x_hat,
            "z_e": z_e,
            "z_q": q["z_q"],
            "indices": q["indices"],
            "quant_loss": q["quant_loss"],
            "perplexity": q["perplexity"],
        }

    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Computes VQ-VAE loss terms.

        Args:
            x: Ground-truth frame batch.
            outputs: Forward outputs from :meth:`forward`.

        Returns:
            Dictionary containing total loss and components.
        """
        recon = self._reconstruction_loss(outputs["x_hat"], x, self.recon_loss_mode)
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
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """Computes reconstruction loss with optional auto mode.

        Args:
            x_hat: Reconstruction tensor.
            x: Ground-truth tensor.
            mode: Loss mode string.

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


class FrameFSQVAE(FrameVQVAE):
    """Finite-scalar-quantized autoencoder for frame vectors ``[B, D]``."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        fsq_levels: int = 8,
        beta: float = 0.25,
        recon_loss_mode: str = "mse",
    ) -> None:
        """Initializes a frame-level FSQ-VAE model.

        Args:
            input_dim: Input frame feature dimension.
            embedding_dim: Latent embedding dimension before quantization.
            hidden_dim: Hidden MLP width.
            fsq_levels: Number of scalar quantization bins.
            beta: Commitment loss weight.
            recon_loss_mode: Reconstruction loss mode, one of
                ``{"auto", "bce", "mse"}``.
        """
        super().__init__(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_embeddings=fsq_levels,
            beta=beta,
            recon_loss_mode=recon_loss_mode,
        )
        self.quantizer = FSQQuantizer(levels=int(fsq_levels), beta=float(beta))
