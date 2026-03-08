"""Core modules for motion-based VQ/FSQ training."""

from .data import DataConfig, create_motion_dataloaders
from .motion_load import (
    MotionDatasetConfig,
    MotionFrameDataset,
    load_motion_feature_sequence,
    resolve_motion_files,
)
from .quantizers import FSQQuantizer, IFSQuantizer, VectorQuantizer
from .vqvae import FrameFSQVAE, FrameIFSQVAE, FrameVQVAE

__all__ = [
    "DataConfig",
    "create_motion_dataloaders",
    "MotionDatasetConfig",
    "MotionFrameDataset",
    "load_motion_feature_sequence",
    "resolve_motion_files",
    "VectorQuantizer",
    "FSQQuantizer",
    "IFSQuantizer",
    "FrameVQVAE",
    "FrameFSQVAE",
    "FrameIFSQVAE",
]
