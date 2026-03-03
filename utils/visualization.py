"""Visualization and reconstruction-saving utilities."""

from __future__ import annotations

import math
from pathlib import Path

import torch


def save_reconstruction_batch(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    output_path: Path,
    image_size: int | None = None,
) -> None:
    """Saves a side-by-side reconstruction artifact.

    The function attempts to save PNG grids with ``torchvision`` if available.
    When unavailable, it falls back to saving raw tensors as ``.pt``.

    Args:
        x: Original tensor with shape ``[batch, D]`` or ``[batch, 1, H, W]``.
        x_hat: Reconstructed tensor with matching shape.
        output_path: Target file path.
        image_size: Optional height/width used to reshape flattened vectors.
            If ``None`` and vectors are square, it is inferred automatically.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_items = min(8, x.shape[0])
    original = x[:num_items].detach().cpu()
    recon = x_hat[:num_items].detach().cpu()

    if original.ndim == 4:
        try:
            from torchvision.utils import save_image

            stacked = torch.cat([original, recon], dim=0)
            save_image(stacked, str(output_path), nrow=num_items)
            return
        except Exception:
            pass

    if image_size is None and original.ndim == 2:
        size = int(math.sqrt(int(original.shape[1])))
        image_size = size if size * size == int(original.shape[1]) else None

    if image_size is not None and original.ndim == 2 and original.shape[1] == image_size * image_size:
        try:
            from torchvision.utils import save_image

            # Alternate original and reconstruction for quick visual inspection.
            stacked = torch.cat([original, recon], dim=0).view(-1, 1, image_size, image_size)
            save_image(stacked, str(output_path), nrow=num_items)
            return
        except Exception:
            pass

    torch.save({"original": original, "reconstruction": recon}, output_path.with_suffix(".pt"))
