"""Quantization modules for VQ-VAE and FSQ-based variants."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Classic vector quantizer with trainable codebook embeddings."""

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25) -> None:
        """Initializes vector quantizer.

        Args:
            num_embeddings: Number of codebook entries.
            embedding_dim: Dimensionality of each code vector.
            beta: Commitment coefficient.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Quantizes flattened latent vectors using nearest codebook entries.

        Args:
            z_e: Encoder output with shape ``[N, embedding_dim]``.

        Returns:
            Dictionary with quantization outputs and losses.
        """
        # Squared Euclidean distance between encoder outputs and code vectors.
        distances = (
            torch.sum(z_e**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * z_e @ self.codebook.weight.t()
        )
        indices = torch.argmin(distances, dim=1)
        z_q = self.codebook(indices)

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        loss = codebook_loss + self.beta * commitment_loss
        z_q_st = z_e + (z_q - z_e).detach()

        one_hot = F.one_hot(indices, num_classes=self.num_embeddings).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            "z_q": z_q_st,
            "indices": indices,
            "quant_loss": loss,
            "perplexity": perplexity,
        }


class FSQQuantizer(nn.Module):
    """Finite Scalar Quantization (FSQ) for latent feature vectors.

    FSQ discretizes each latent scalar independently into a fixed number of
    levels in ``[-1, 1]`` and applies straight-through estimation.
    """

    def __init__(self, levels: int | Iterable[int] = 8, beta: float = 0.25) -> None:
        """Initializes FSQ quantizer.

        Args:
            levels: Number of quantization bins. If iterable, first value is used.
            beta: Commitment coefficient.
        """
        super().__init__()
        if isinstance(levels, int):
            self.levels = int(levels)
        else:
            level_list = list(levels)
            self.levels = int(level_list[0])
        if self.levels < 2:
            raise ValueError("FSQ levels must be >= 2.")
        self.beta = float(beta)
        self.register_buffer("grid", torch.linspace(-1.0, 1.0, self.levels))

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Quantizes flattened vectors by nearest scalar levels.

        Args:
            z_e: Encoder outputs with shape ``[N, D]``.

        Returns:
            Dictionary with quantized tensor and training losses.
        """
        z_norm = torch.tanh(z_e)
        distances = torch.abs(z_norm.unsqueeze(-1) - self.grid.view(1, 1, -1))
        indices = torch.argmin(distances, dim=-1)
        z_q = self.grid[indices]

        codebook_loss = F.mse_loss(z_q, z_norm.detach())
        commitment_loss = F.mse_loss(z_norm, z_q.detach())
        loss = codebook_loss + self.beta * commitment_loss
        z_q_st = z_norm + (z_q - z_norm).detach()

        # Use the first few dimensions to form a lightweight pseudo-code ID.
        code_dims = min(indices.shape[1], 4)
        bases = torch.tensor(
            [self.levels**k for k in range(code_dims)],
            device=indices.device,
            dtype=indices.dtype,
        )
        mixed_indices = torch.sum(indices[:, :code_dims] * bases.unsqueeze(0), dim=1)
        unique_codes = torch.unique(mixed_indices).numel()
        perplexity = torch.tensor(
            float(unique_codes) / float(max(mixed_indices.numel(), 1)),
            device=z_e.device,
            dtype=z_e.dtype,
        )

        return {"z_q": z_q_st, "indices": mixed_indices, "quant_loss": loss, "perplexity": perplexity}
