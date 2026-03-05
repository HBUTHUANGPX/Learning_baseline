"""Sequence-oriented models for temporal reconstruction experiments."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import FSQQuantizer


class SequenceFSQModel(nn.Module):
    """FSQ autoencoder with 1D convolutions for sequence reconstruction.

    This model targets temporal inputs shaped as ``[B, T, D]`` where:
    - ``B`` is batch size
    - ``T`` is sequence length
    - ``D`` is per-frame feature dimension

    It also supports frame-wise tensors ``[B, D]`` by internally promoting them
    to ``T=1`` sequences.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 16,
        hidden_channels: int = 64,
        fsq_levels: int = 8,
        beta: float = 0.25,
        recon_loss_mode: str = "auto",
    ) -> None:
        """Initializes a temporal FSQ model.

        Args:
            input_dim: Feature dimension per frame.
            embedding_dim: Latent channel width before quantization.
            hidden_channels: Hidden channel width of Conv1d backbone.
            fsq_levels: Number of scalar quantization bins.
            beta: Commitment loss coefficient.
            recon_loss_mode: Reconstruction loss mode in ``{"auto","bce","mse"}``.
        """
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.hidden_channels = int(hidden_channels)
        self.recon_loss_mode = str(recon_loss_mode)

        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                self.hidden_channels, self.embedding_dim, kernel_size=3, padding=1
            ),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_channels, self.input_dim, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.quantizer = FSQQuantizer(levels=fsq_levels, beta=beta)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs one forward pass.

        Args:
            x: Input tensor with shape ``[B, T, D]`` or ``[B, D]``.

        Returns:
            Dictionary containing reconstructions and quantization stats.

        Raises:
            ValueError: If input rank is not 2 or 3.
        """
        was_frame_input = x.ndim == 2
        if was_frame_input:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(
                f"SequenceFSQModel expects input shape [B,T,D] or [B,D], got {tuple(x.shape)}."
            )

        # Conv1d uses channel-first layout: [B, D, T].
        x_c = x.transpose(1, 2)
        z_e_map = self.encoder(x_c)
        z_e_flat = z_e_map.transpose(1, 2).reshape(-1, self.embedding_dim)
        q = self.quantizer(z_e_flat)
        z_q_map = q["z_q"].view(x.shape[0], x.shape[1], self.embedding_dim).transpose(1, 2)
        x_hat = self.decoder(z_q_map).transpose(1, 2)

        if was_frame_input:
            x_hat = x_hat.squeeze(1)
            z_e_map = z_e_map.squeeze(-1)
            z_q_map = z_q_map.squeeze(-1)

        return {
            "x_hat": x_hat,
            "z_e": z_e_map,
            "z_q": z_q_map,
            "indices": q["indices"],
            "quant_loss": q["quant_loss"],
            "perplexity": q["perplexity"],
        }

    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Computes FSQ sequence reconstruction objective.

        Args:
            x: Target tensor, shape ``[B,T,D]`` or ``[B,D]``.
            outputs: Forward output dictionary from ``forward``.
            mask: Optional validity mask for variable-length sequences with shape
                ``[B,T]``. Only used when ``x`` is 3D.

        Returns:
            Dictionary with scalar loss terms.
        """
        recon = self._reconstruction_loss(
            outputs["x_hat"],
            x,
            mode=self.recon_loss_mode,
            mask=mask,
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
        x: torch.Tensor,
        mode: str = "auto",
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes masked/unmasked reconstruction loss.

        Args:
            x_hat: Reconstruction tensor.
            x: Ground-truth tensor.
            mode: One of ``{"auto","bce","mse"}``.
            mask: Optional sequence validity mask ``[B,T]``.

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
            per_elem = F.binary_cross_entropy(x_hat, x, reduction="none")
        else:
            per_elem = F.mse_loss(x_hat, x, reduction="none")

        if mask is not None and x.ndim == 3:
            # Apply time-step mask on sequence losses and keep per-batch scaling.
            per_elem = per_elem * mask.unsqueeze(-1).to(per_elem.dtype)

        return per_elem.sum() / max(int(x.shape[0]), 1)
