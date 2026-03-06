"""Frame-level VQ-VAE and FSQ-VAE with decoder history conditioning."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import FSQQuantizer, VectorQuantizer


class FrameVQVAE(nn.Module):
    """Vector-quantized autoencoder with conditional decoder.

    Encoder input is a flattened temporal context window. Decoder receives both
    quantized latent and flattened history-condition vectors.
    """

    def __init__(
        self,
        encoder_input_dim: int,
        decoder_condition_dim: int,
        target_dim: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        num_embeddings: int = 512,
        beta: float = 0.25,
        recon_loss_mode: str = "mse",
    ) -> None:
        """Initializes a conditional frame-level VQ-VAE model.

        Args:
            encoder_input_dim: Encoder input feature dimension.
            decoder_condition_dim: Decoder condition feature dimension.
            target_dim: Reconstruction target dimension.
            embedding_dim: Latent embedding dimension before quantization.
            hidden_dim: Hidden MLP width.
            num_embeddings: Number of VQ codebook vectors.
            beta: Commitment loss weight.
            recon_loss_mode: Reconstruction loss mode in
                ``{"auto", "bce", "mse"}``.
        """
        super().__init__()
        self.encoder_input_dim = int(encoder_input_dim)
        self.decoder_condition_dim = int(decoder_condition_dim)
        self.target_dim = int(target_dim)
        self.embedding_dim = int(embedding_dim)
        self.recon_loss_mode = str(recon_loss_mode)

        self.encoder = nn.Sequential(
            nn.Linear(self.encoder_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.embedding_dim),
        )

        decoder_in_dim = self.embedding_dim + self.decoder_condition_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.target_dim),
        )

        self.quantizer = VectorQuantizer(
            num_embeddings=int(num_embeddings),
            embedding_dim=self.embedding_dim,
            beta=float(beta),
        )

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_condition: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Runs one forward pass.

        Args:
            encoder_input: Encoder input tensor with shape ``[B, D_enc]``.
            decoder_condition: Decoder condition tensor with shape ``[B, D_cond]``.

        Returns:
            Dictionary containing reconstruction and quantization outputs.
        """
        z_e = self.encoder(encoder_input)
        q = self.quantizer(z_e)
        decoder_input = torch.cat([q["z_q"], decoder_condition], dim=1)
        x_hat = self.decoder(decoder_input)
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
        target: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Computes VQ-VAE loss terms.

        Args:
            target: Reconstruction target tensor.
            outputs: Forward outputs from :meth:`forward`.

        Returns:
            Dictionary containing total loss and components.
        """
        recon = self._reconstruction_loss(
            outputs["x_hat"],
            target,
            self.recon_loss_mode,
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
        x_hat: torch.Tensor,
        target: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """Computes reconstruction loss with optional auto mode.

        Args:
            x_hat: Reconstruction tensor.
            target: Ground-truth tensor.
            mode: Loss mode string.

        Returns:
            Batch-averaged reconstruction loss.
        """
        mode = mode.lower().strip()
        if mode not in {"auto", "bce", "mse"}:
            raise ValueError(f"Unsupported reconstruction loss mode: {mode}")
        if mode == "auto":
            target_in_range = bool(
                torch.logical_and(target >= 0.0, target <= 1.0).all().item()
            )
            xhat_in_range = bool(
                torch.logical_and(x_hat >= 0.0, x_hat <= 1.0).all().item()
            )
            mode = "bce" if (target_in_range and xhat_in_range) else "mse"

        if mode == "bce":
            return F.binary_cross_entropy(x_hat, target, reduction="sum") / target.shape[0]
        return F.mse_loss(x_hat, target, reduction="sum") / target.shape[0]


class FrameFSQVAE(FrameVQVAE):
    """Finite-scalar-quantized autoencoder with conditional decoder."""

    def __init__(
        self,
        encoder_input_dim: int,
        decoder_condition_dim: int,
        target_dim: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        fsq_levels: int = 8,
        beta: float = 0.25,
        recon_loss_mode: str = "mse",
    ) -> None:
        """Initializes a conditional frame-level FSQ-VAE model.

        Args:
            encoder_input_dim: Encoder input feature dimension.
            decoder_condition_dim: Decoder condition feature dimension.
            target_dim: Reconstruction target dimension.
            embedding_dim: Latent embedding dimension before quantization.
            hidden_dim: Hidden MLP width.
            fsq_levels: Number of scalar quantization bins.
            beta: Commitment loss weight.
            recon_loss_mode: Reconstruction loss mode in
                ``{"auto", "bce", "mse"}``.
        """
        super().__init__(
            encoder_input_dim=encoder_input_dim,
            decoder_condition_dim=decoder_condition_dim,
            target_dim=target_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_embeddings=fsq_levels,
            beta=beta,
            recon_loss_mode=recon_loss_mode,
        )
        self.quantizer = FSQQuantizer(levels=int(fsq_levels), beta=float(beta))
