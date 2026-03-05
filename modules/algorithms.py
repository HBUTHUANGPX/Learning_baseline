"""Algorithm-term abstractions and registries for training orchestration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Protocol, Tuple

import torch

from .configs import ConvVAEConfig, ImageConfig, MLPVAEConfig, VQVAEConfig
from .frame_models import FrameConvVAE, FrameFSQVAE, FrameVQVAE
from .registry import Registry
from .sequence_models import SequenceFSQModel
from .vae import BetaVAE, ConvVAE, VanillaVAE
from .vqvae import FSQVAE, VQVAE


class AlgorithmTerm(Protocol):
    """Unified algorithm interface consumed by the training manager."""

    model: torch.nn.Module
    metric_keys: Tuple[str, ...]
    expects_image_input: bool
    expected_input_ndims: Tuple[int, ...]

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
    expected_input_ndims: Tuple[int, ...]


class VAEAlgorithmTerm:
    """Default algorithm term for VAE-family models with ``loss_function`` API."""

    def __init__(
        self,
        model: torch.nn.Module,
        metric_keys: Tuple[str, ...],
        expects_image_input: bool,
        expected_input_ndims: Tuple[int, ...] = (2,),
    ) -> None:
        """Initializes VAE algorithm term.

        Args:
            model: VAE-like model instance.
            metric_keys: Ordered metric names to aggregate.
            expects_image_input: Whether model expects image-shaped inputs.
            expected_input_ndims: Allowed ranks for model inputs.
        """
        self.model = model
        self.metric_keys = metric_keys
        self.expects_image_input = expects_image_input
        self.expected_input_ndims = expected_input_ndims

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
        mask = extract_algorithm_mask(batch)
        if mask is not None:
            mask = mask.to(device)
        outputs = self.model(x)
        if mask is not None:
            try:
                losses = self.model.loss_function(x, outputs, mask=mask)
            except TypeError:
                losses = self.model.loss_function(x, outputs)
        else:
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


def extract_algorithm_mask(batch: torch.Tensor | tuple | dict) -> torch.Tensor | None:
    """Extracts optional sequence mask from protocol batch.

    Args:
        batch: Tensor batch, tuple batch, or protocol dictionary.

    Returns:
        Sequence mask tensor if present, otherwise ``None``.
    """
    if not isinstance(batch, dict):
        return None
    meta = batch.get("meta")
    if not isinstance(meta, dict):
        return None
    mask = meta.get("policy_mask")
    if isinstance(mask, torch.Tensor):
        return mask
    return None


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


def _is_motion_frame_mode(args: argparse.Namespace) -> bool:
    """Checks whether current run uses motion_mimic frame-wise batches."""
    return args.dataset == "motion_mimic" and not bool(args.motion_as_sequence)


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
        expected_input_ndims=(2,),
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
        expected_input_ndims=(2,),
    )


@MODEL_REGISTRY.register("conv")
def build_conv_model(args: argparse.Namespace) -> ModelSpec:
    """Builds ConvVAE model spec from args."""
    if _is_motion_frame_mode(args):
        conv_channels = _parse_int_tuple(args.conv_channels)
        hidden_channels = conv_channels[0] if len(conv_channels) > 0 else 64
        model = FrameConvVAE(
            input_dim=int(args.input_dim),
            latent_dim=int(args.latent_dim),
            hidden_channels=int(hidden_channels),
            bottleneck_dim=int(args.conv_bottleneck_dim),
            recon_loss_mode=str(args.recon_loss_mode),
        )
        return ModelSpec(
            model=model,
            metric_keys=("loss", "recon_loss", "kl_loss"),
            expects_image_input=False,
            expected_input_ndims=(2,),
        )

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
        expected_input_ndims=(4,),
    )


@MODEL_REGISTRY.register("vq")
def build_vq_model(args: argparse.Namespace) -> ModelSpec:
    """Builds VQVAE model spec from args."""
    if _is_motion_frame_mode(args):
        conv_channels = _parse_int_tuple(args.conv_channels)
        hidden_channels = conv_channels[0] if len(conv_channels) > 0 else 64
        model = FrameVQVAE(
            input_dim=int(args.input_dim),
            embedding_dim=int(args.latent_dim),
            hidden_channels=int(hidden_channels),
            num_embeddings=int(args.num_embeddings),
            beta=float(args.beta),
            recon_loss_mode=str(args.recon_loss_mode),
        )
        return ModelSpec(
            model=model,
            metric_keys=("loss", "recon_loss", "quant_loss", "perplexity"),
            expects_image_input=False,
            expected_input_ndims=(2,),
        )

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
        expected_input_ndims=(4,),
    )


@MODEL_REGISTRY.register("fsq")
def build_fsq_model(args: argparse.Namespace) -> ModelSpec:
    """Builds FSQVAE model spec from args."""
    if _is_motion_frame_mode(args):
        conv_channels = _parse_int_tuple(args.conv_channels)
        hidden_channels = conv_channels[0] if len(conv_channels) > 0 else 64
        model = FrameFSQVAE(
            input_dim=int(args.input_dim),
            embedding_dim=int(args.latent_dim),
            hidden_channels=int(hidden_channels),
            fsq_levels=int(args.fsq_levels),
            beta=float(args.beta),
            recon_loss_mode=str(args.recon_loss_mode),
        )
        return ModelSpec(
            model=model,
            metric_keys=("loss", "recon_loss", "quant_loss", "perplexity"),
            expects_image_input=False,
            expected_input_ndims=(2,),
        )

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
    is_sequence_dataset = args.dataset == "random_sequence" or (
        args.dataset == "motion_mimic" and bool(args.motion_as_sequence)
    )
    if is_sequence_dataset:
        conv_channels = _parse_int_tuple(args.conv_channels)
        hidden_channels = conv_channels[0] if len(conv_channels) > 0 else 64
        model = SequenceFSQModel(
            input_dim=int(args.sequence_feature_dim),
            embedding_dim=int(args.latent_dim),
            hidden_channels=int(hidden_channels),
            fsq_levels=int(args.fsq_levels),
            beta=float(args.beta),
            recon_loss_mode=str(args.recon_loss_mode),
        )
        return ModelSpec(
            model=model,
            metric_keys=("loss", "recon_loss", "quant_loss", "perplexity"),
            expects_image_input=False,
            expected_input_ndims=(3, 2),
        )

    return ModelSpec(
        model=FSQVAE(recon_loss_mode=cfg.recon_loss_mode, config=cfg),
        metric_keys=("loss", "recon_loss", "quant_loss", "perplexity"),
        expects_image_input=True,
        expected_input_ndims=(4,),
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
        expected_input_ndims=spec.expected_input_ndims,
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
