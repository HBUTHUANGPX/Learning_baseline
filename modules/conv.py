"""Convolutional encoder and decoder blocks for image VAEs."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class ConvGaussianEncoder(nn.Module):
    """Convolutional Gaussian encoder for 28x28 grayscale images.

    The encoder downsamples ``[B, 1, 28, 28]`` into compact feature maps and
    projects them into latent Gaussian parameters ``(mu, logvar)``.
    """

    def __init__(self, latent_dim: int) -> None:
        """Initializes the convolutional Gaussian encoder.

        Args:
            latent_dim: Target latent space dimension.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, 256)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes posterior Gaussian parameters.

        Args:
            x: Image tensor of shape ``[batch, 1, 28, 28]``.

        Returns:
            Tuple ``(mu, logvar)``, each with shape ``[batch, latent_dim]``.
        """
        h = self.features(x)
        h = self.flatten(h)
        h = torch.relu(self.fc(h))
        return self.mu(h), self.logvar(h)


class ConvDecoder(nn.Module):
    """Convolutional decoder that reconstructs 28x28 grayscale images."""

    def __init__(self, latent_dim: int) -> None:
        """Initializes the convolutional decoder.

        Args:
            latent_dim: Latent space dimension.
        """
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vectors into images.

        Args:
            z: Latent tensor with shape ``[batch, latent_dim]``.

        Returns:
            Reconstructed image tensor with shape ``[batch, 1, 28, 28]``.
        """
        h = torch.relu(self.fc(z))
        h = h.view(z.shape[0], 64, 7, 7)
        return self.deconv(h)


class VQConvEncoder(nn.Module):
    """Convolutional encoder used by VQ-VAE style models."""

    def __init__(self, embedding_dim: int) -> None:
        """Initializes the encoder.

        Args:
            embedding_dim: Channel dimension of latent feature map.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes images into latent feature maps.

        Args:
            x: Image tensor with shape ``[batch, 1, 28, 28]``.

        Returns:
            Latent feature map with shape ``[batch, embedding_dim, 7, 7]``.
        """
        return self.net(x)


class VQConvDecoder(nn.Module):
    """Convolutional decoder used by VQ-VAE style models."""

    def __init__(self, embedding_dim: int) -> None:
        """Initializes the decoder.

        Args:
            embedding_dim: Channel dimension of latent feature map.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decodes quantized maps into reconstructed images.

        Args:
            z_q: Quantized map with shape ``[batch, embedding_dim, 7, 7]``.

        Returns:
            Reconstructed image tensor ``[batch, 1, 28, 28]``.
        """
        return self.net(z_q)


def flatten_spatial(z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Flattens spatial latent maps for vector quantization.

    Args:
        z: Latent map with shape ``[batch, channels, height, width]``.

    Returns:
        Tuple of:
        - Flattened latent tensor ``[batch * height * width, channels]``.
        - Shape metadata for restoring original layout.
    """
    b, c, h, w = z.shape
    z_perm = z.permute(0, 2, 3, 1).contiguous()
    flat = z_perm.view(-1, c)
    return flat, {"b": b, "c": c, "h": h, "w": w}


def unflatten_spatial(flat: torch.Tensor, meta: Dict[str, int]) -> torch.Tensor:
    """Restores flattened latent vectors to image feature-map layout.

    Args:
        flat: Flattened latent tensor ``[batch * height * width, channels]``.
        meta: Shape metadata from ``flatten_spatial``.

    Returns:
        Tensor with shape ``[batch, channels, height, width]``.
    """
    z = flat.view(meta["b"], meta["h"], meta["w"], meta["c"])
    return z.permute(0, 3, 1, 2).contiguous()
