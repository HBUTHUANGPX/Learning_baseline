"""Reusable neural network modules for VAE experiments."""

from .activations import ActivationFactory
from .algorithms import (
    ALGORITHM_REGISTRY,
    MODEL_REGISTRY,
    AlgorithmTerm,
    ModelSpec,
    VAEAlgorithmTerm,
    build_algorithm_term,
)
from .configs import ConvVAEConfig, ImageConfig, MLPVAEConfig, VQVAEConfig
from .conv import ConvDecoder, ConvGaussianEncoder, VQConvDecoder, VQConvEncoder
from .data import DATASET_REGISTRY, create_dataloader
from .decoders import MLPDecoder
from .encoders import MLPGaussianEncoder
from .frame_models import FrameConvVAE, FrameFSQVAE, FrameVQVAE
from .observations import (
    ObsGroupCfg,
    ObsTermCfg,
    ObservationManager,
    ObservationsCfg,
)
from .quantizers import FSQQuantizer, VectorQuantizer
from .registry import Registry
from .sequence_models import SequenceFSQModel
from .vae import BetaVAE, ConvVAE, VanillaVAE
from .vqvae import FSQVAE, VQVAE

__all__ = [
    "ActivationFactory",
    "Registry",
    "MODEL_REGISTRY",
    "ALGORITHM_REGISTRY",
    "DATASET_REGISTRY",
    "AlgorithmTerm",
    "ModelSpec",
    "VAEAlgorithmTerm",
    "build_algorithm_term",
    "ImageConfig",
    "MLPVAEConfig",
    "ConvVAEConfig",
    "VQVAEConfig",
    "ConvDecoder",
    "ConvGaussianEncoder",
    "VQConvDecoder",
    "VQConvEncoder",
    "create_dataloader",
    "ObsTermCfg",
    "ObsGroupCfg",
    "ObservationsCfg",
    "ObservationManager",
    "MLPDecoder",
    "MLPGaussianEncoder",
    "FrameConvVAE",
    "FrameVQVAE",
    "FrameFSQVAE",
    "VectorQuantizer",
    "FSQQuantizer",
    "SequenceFSQModel",
    "BetaVAE",
    "ConvVAE",
    "VanillaVAE",
    "VQVAE",
    "FSQVAE",
]
