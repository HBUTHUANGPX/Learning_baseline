"""VQ-VAE and FSQ-VAE model definitions."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import VQConvDecoder, VQConvEncoder, flatten_spatial, unflatten_spatial
from .quantizers import FSQQuantizer, VectorQuantizer


class VQVAE(nn.Module):
    """Vector-Quantized VAE for 28x28 grayscale images."""

    def __init__(
        self,
        embedding_dim: int = 16,
        num_embeddings: int = 128,
        beta: float = 0.25,
    ) -> None:
        """Initializes a VQ-VAE model.

        Args:
            embedding_dim: Channel size of latent feature maps.
            num_embeddings: Number of codebook vectors.
            beta: Commitment coefficient for quantization.
        """
        super().__init__()
        self.encoder = VQConvEncoder(embedding_dim=embedding_dim)
        self.decoder = VQConvDecoder(embedding_dim=embedding_dim)
        self.quantizer = VectorQuantizer(
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

    def loss_function(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Computes VQ-VAE loss.

        Args:
            x: Ground-truth input image batch.
            outputs: Forward outputs from ``self.forward``.

        Returns:
            Dictionary with scalar loss terms.
        """
        recon = F.binary_cross_entropy(outputs["x_hat"], x, reduction="sum") / x.shape[0]
        quant = outputs["quant_loss"]
        total = recon + quant
        return {
            "loss": total,
            "recon_loss": recon,
            "quant_loss": quant,
            "perplexity": outputs["perplexity"],
        }


class FSQVAE(VQVAE):
    """FSQ-VAE using finite scalar quantization instead of vector codebook."""

    def __init__(
        self,
        embedding_dim: int = 16,
        fsq_levels: int = 8,
        beta: float = 0.25,
    ) -> None:
        """Initializes FSQ-VAE.

        Args:
            embedding_dim: Channel size of latent fea ture maps.
            fsq_levels: Number of scalar bins in FSQ.
            beta: Commitment coefficient for FSQ.
        """
        nn.Module.__init__(self)
        self.encoder = VQConvEncoder(embedding_dim=embedding_dim)
        self.decoder = VQConvDecoder(embedding_dim=embedding_dim)
        self.quantizer = FSQQuantizer(levels=fsq_levels, beta=beta)
