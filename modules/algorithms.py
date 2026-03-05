"""Algorithm-term abstractions and registries for training orchestration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Protocol, Tuple

import torch

from .configs import ConvVAEConfig, ImageConfig, MLPVAEConfig, VQVAEConfig
from .registry import Registry
from .vae import BetaVAE, ConvVAE, VanillaVAE
from .vqvae import FSQVAE, VQVAE


class AlgorithmTerm(Protocol):
    """Unified algorithm interface consumed by the training manager."""

    model: torch.nn.Module
    metric_keys: Tuple[str, ...]
    expects_image_input: bool

    def compute(
        self,
        batch: torch.Tensor | tuple | dict,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Runs one forward-loss computation from a raw dataloader batch.

        Args:
            batch: Raw batch from dataloader.
            device: Computation device.

        Returns:
            Tuple of ``(inputs, outputs, losses)``.
        """


@dataclass(frozen=True)
class ModelSpec:
    """Model specification returned by model builders."""

    model: torch.nn.Module
    metric_keys: Tuple[str, ...]
    expects_image_input: bool


class VAEAlgorithmTerm:
    """Default algorithm term for VAE-family models with ``loss_function`` API."""

    def __init__(
        self,
        model: torch.nn.Module,
        metric_keys: Tuple[str, ...],
        expects_image_input: bool,
    ) -> None:
        """Initializes VAE algorithm term.

        Args:
            model: VAE-like model instance.
            metric_keys: Ordered metric names to aggregate.
            expects_image_input: Whether model expects image-shaped inputs.
        """
        self.model = model
        self.metric_keys = metric_keys
        self.expects_image_input = expects_image_input

    def compute(
        self,
        batch: torch.Tensor | tuple | dict,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Computes forward outputs and losses from raw batch.

        Args:
            batch: Raw dataloader batch.
            device: Computation device.

        Returns:
            Tuple of ``(inputs, outputs, losses)``.
        """
        x = extract_algorithm_inputs(batch).to(device)
        outputs = self.model(x)
        losses = self.model.loss_function(x, outputs)
        return x, outputs, losses


def extract_algorithm_inputs(batch: torch.Tensor | tuple | dict) -> torch.Tensor:
    """Extracts primary input tensor from protocol-compliant batch structures.

    Args:
        batch: Tensor batch, tuple batch, or observation protocol dictionary.

    Returns:
        Primary input tensor.
    """
    if isinstance(batch, dict):
        if "obs" in batch and isinstance(batch["obs"], dict):
            obs = batch["obs"]
            if "policy" in obs:
                return obs["policy"]
            return next(iter(obs.values()))
        if "x" in batch and isinstance(batch["x"], torch.Tensor):
            return batch["x"]
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value
        raise ValueError("Unsupported dictionary batch: no tensor-like inputs found.")
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


MODEL_REGISTRY: Registry = Registry("model")
ALGORITHM_REGISTRY: Registry = Registry("algorithm")


def _build_image_cfg(args: argparse.Namespace) -> ImageConfig:
    """Builds image configuration from CLI namespace.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Image configuration object.
    """
    return ImageConfig(
        channels=args.image_channels,
        height=args.image_height,
        width=args.image_width,
    )


def _parse_int_tuple(raw: str) -> Tuple[int, ...]:
    """Parses comma-separated integer list string."""
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


@MODEL_REGISTRY.register("vanilla")
def build_vanilla_model(args: argparse.Namespace) -> ModelSpec:
    """Builds VanillaVAE model spec from args."""
    cfg = MLPVAEConfig(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        hidden_dims=_parse_int_tuple(args.hidden_dims),
        activation=args.activation,
        beta=args.beta,
        recon_loss_mode=args.recon_loss_mode,
    )
    model = VanillaVAE(
        input_dim=cfg.input_dim,
        latent_dim=cfg.latent_dim,
        recon_loss_mode=cfg.recon_loss_mode,
        config=cfg,
    )
    return ModelSpec(
        model=model,
        metric_keys=("loss", "recon_loss", "kl_loss"),
        expects_image_input=False,
    )


@MODEL_REGISTRY.register("beta")
def build_beta_model(args: argparse.Namespace) -> ModelSpec:
    """Builds BetaVAE model spec from args."""
    cfg = MLPVAEConfig(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        hidden_dims=_parse_int_tuple(args.hidden_dims),
        activation=args.activation,
        beta=args.beta,
        recon_loss_mode=args.recon_loss_mode,
    )
    model = BetaVAE(
        input_dim=cfg.input_dim,
        latent_dim=cfg.latent_dim,
        beta=cfg.beta,
        recon_loss_mode=cfg.recon_loss_mode,
        config=cfg,
    )
    return ModelSpec(
        model=model,
        metric_keys=("loss", "recon_loss", "kl_loss"),
        expects_image_input=False,
    )


@MODEL_REGISTRY.register("conv")
def build_conv_model(args: argparse.Namespace) -> ModelSpec:
    """Builds ConvVAE model spec from args."""
    cfg = ConvVAEConfig(
        image=_build_image_cfg(args),
        latent_dim=args.latent_dim,
        encoder_channels=_parse_int_tuple(args.conv_channels),
        bottleneck_dim=args.conv_bottleneck_dim,
        recon_loss_mode=args.recon_loss_mode,
    )
    return ModelSpec(
        model=ConvVAE(recon_loss_mode=cfg.recon_loss_mode, config=cfg),
        metric_keys=("loss", "recon_loss", "kl_loss"),
        expects_image_input=True,
    )


@MODEL_REGISTRY.register("vq")
def build_vq_model(args: argparse.Namespace) -> ModelSpec:
    """Builds VQVAE model spec from args."""
    cfg = VQVAEConfig(
        image=_build_image_cfg(args),
        embedding_dim=args.latent_dim,
        encoder_channels=_parse_int_tuple(args.conv_channels),
        decoder_channels=_parse_int_tuple(args.vq_decoder_channels),
        num_embeddings=args.num_embeddings,
        beta=args.beta,
        fsq_levels=args.fsq_levels,
        recon_loss_mode=args.recon_loss_mode,
    )
    return ModelSpec(
        model=VQVAE(recon_loss_mode=cfg.recon_loss_mode, config=cfg),
        metric_keys=("loss", "recon_loss", "quant_loss", "perplexity"),
        expects_image_input=True,
    )


@MODEL_REGISTRY.register("fsq")
def build_fsq_model(args: argparse.Namespace) -> ModelSpec:
    """Builds FSQVAE model spec from args."""
    cfg = VQVAEConfig(
        image=_build_image_cfg(args),
        embedding_dim=args.latent_dim,
        encoder_channels=_parse_int_tuple(args.conv_channels),
        decoder_channels=_parse_int_tuple(args.vq_decoder_channels),
        num_embeddings=args.num_embeddings,
        beta=args.beta,
        fsq_levels=args.fsq_levels,
        recon_loss_mode=args.recon_loss_mode,
    )
    return ModelSpec(
        model=FSQVAE(recon_loss_mode=cfg.recon_loss_mode, config=cfg),
        metric_keys=("loss", "recon_loss", "quant_loss", "perplexity"),
        expects_image_input=True,
    )


@ALGORITHM_REGISTRY.register("vae")
def build_vae_algorithm_term(args: argparse.Namespace) -> AlgorithmTerm:
    """Builds VAE algorithm term by composing model registry output.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Algorithm term instance.
    """
    model_builder = MODEL_REGISTRY.get(args.model)
    spec: ModelSpec = model_builder(args)
    return VAEAlgorithmTerm(
        model=spec.model,
        metric_keys=spec.metric_keys,
        expects_image_input=spec.expects_image_input,
    )


def build_algorithm_term(args: argparse.Namespace) -> AlgorithmTerm:
    """Builds algorithm term from algorithm registry.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Algorithm term selected by ``args.algorithm``.
    """
    algorithm_builder = ALGORITHM_REGISTRY.get(args.algorithm)
    return algorithm_builder(args)
