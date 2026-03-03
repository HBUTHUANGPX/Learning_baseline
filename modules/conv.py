"""Convolutional encoder and decoder blocks for image VAEs."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn


class ConvGaussianEncoder(nn.Module):
    """Convolutional Gaussian encoder for 28x28 grayscale images.

    The encoder downsamples ``[B, 1, 28, 28]`` into compact feature maps and
    projects them into latent Gaussian parameters ``(mu, logvar)``.
    """

    def __init__(
        self,
        latent_dim: int,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        conv_channels: Sequence[int] = (32, 64),
        bottleneck_dim: int = 256,
    ) -> None:
        """Initializes the convolutional Gaussian encoder.

        Args:
            latent_dim: Target latent space dimension.
            image_shape: Input image shape ``(C, H, W)``.
            conv_channels: Encoder channel plan for strided conv layers.
            bottleneck_dim: Fully-connected bottleneck width.
        """
        super().__init__()
        if len(conv_channels) < 2:
            raise ValueError("conv_channels must contain at least 2 values.")
        c_in = image_shape[0]
        blocks = []
        for c_out in conv_channels:
            blocks.append(nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            blocks.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        with torch.no_grad():
            feature = self.features(torch.zeros(1, *image_shape))
        self.feature_shape = (feature.shape[1], feature.shape[2], feature.shape[3])
        self.flat_dim = int(feature.numel())
        self.fc = nn.Linear(self.flat_dim, bottleneck_dim)
        self.mu = nn.Linear(bottleneck_dim, latent_dim)
        self.logvar = nn.Linear(bottleneck_dim, latent_dim)

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

    def __init__(
        self,
        latent_dim: int,
        feature_shape: Tuple[int, int, int],
        output_channels: int = 1,
        decoder_channels: Sequence[int] = (32,),
    ) -> None:
        """Initializes the convolutional decoder.

        Args:
            latent_dim: Latent space dimension.
            feature_shape: Encoder feature shape ``(C, H, W)`` before flattening.
            output_channels: Number of output image channels.
            decoder_channels: Intermediate channels for transposed-conv blocks.
        """
        super().__init__()
        self.feature_shape = feature_shape
        in_c, h, w = feature_shape
        self.fc = nn.Linear(latent_dim, in_c * h * w)
        layers = []
        c_in = in_c
        for c_out in decoder_channels:
            layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        layers.append(nn.ConvTranspose2d(c_in, output_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Sigmoid())
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vectors into images.

        Args:
            z: Latent tensor with shape ``[batch, latent_dim]``.

        Returns:
            Reconstructed image tensor with shape ``[batch, 1, 28, 28]``.
        """
        h = torch.relu(self.fc(z))
        h = h.view(z.shape[0], *self.feature_shape)
        return self.deconv(h)


class VQConvEncoder(nn.Module):
    """Convolutional encoder used by VQ-VAE style models."""

    def __init__(
        self,
        embedding_dim: int,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        conv_channels: Sequence[int] = (32, 64),
    ) -> None:
        """Initializes the encoder.

        Args:
            embedding_dim: Channel dimension of latent feature map.
            image_shape: Input image shape ``(C, H, W)``.
            conv_channels: Channel plan for downsampling layers.
        """
        super().__init__()
        if len(conv_channels) < 2:
            raise ValueError("conv_channels must contain at least 2 values.")
        c_in = image_shape[0]
        layers = []
        for c_out in conv_channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        layers.append(nn.Conv2d(c_in, embedding_dim, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            feature = self.net(torch.zeros(1, *image_shape))
        self.feature_shape = (feature.shape[1], feature.shape[2], feature.shape[3])

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

    def __init__(
        self,
        embedding_dim: int,
        output_channels: int = 1,
        decoder_channels: Sequence[int] = (64, 32),
    ) -> None:
        """Initializes the decoder.

        Args:
            embedding_dim: Channel dimension of latent feature map.
            output_channels: Number of output image channels.
            decoder_channels: Intermediate channels for upsampling layers.
        """
        super().__init__()
        if len(decoder_channels) < 2:
            raise ValueError("decoder_channels must contain at least 2 values.")
        layers = [
            nn.Conv2d(embedding_dim, decoder_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ]
        for idx in range(len(decoder_channels) - 1):
            layers.append(
                nn.ConvTranspose2d(
                    decoder_channels[idx],
                    decoder_channels[idx + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.ConvTranspose2d(
                decoder_channels[-1],
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

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
