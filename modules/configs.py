"""Configuration objects for model architecture and image shapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ImageConfig:
    """Image shape configuration.

    Attributes:
        channels: Number of image channels.
        height: Image height.
        width: Image width.
    """

    channels: int = 1
    height: int = 28
    width: int = 28


@dataclass(frozen=True)
class MLPVAEConfig:
    """Configuration for MLP-based VAE variants."""

    input_dim: int = 784
    latent_dim: int = 16
    hidden_dims: Tuple[int, ...] = (256, 128)
    activation: str = "relu"
    beta: float = 4.0


@dataclass(frozen=True)
class ConvVAEConfig:
    """Configuration for convolutional VAE."""

    image: ImageConfig = ImageConfig()
    latent_dim: int = 16
    encoder_channels: Tuple[int, ...] = (32, 64)
    bottleneck_dim: int = 256


@dataclass(frozen=True)
class VQVAEConfig:
    """Configuration for VQ-VAE and FSQ-VAE models."""

    image: ImageConfig = ImageConfig()
    embedding_dim: int = 16
    encoder_channels: Tuple[int, ...] = (32, 64)
    decoder_channels: Tuple[int, ...] = (64, 32)
    num_embeddings: int = 128
    beta: float = 0.25
    fsq_levels: int = 8
