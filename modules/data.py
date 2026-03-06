"""Data pipeline for frame-level motion VQ/FSQ training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset

from utils.load_motion_file import collect_npz_paths

from .motion_load import MotionDatasetConfig, MotionFrameDataset


@dataclass(frozen=True)
class DataConfig:
    """Configuration for motion dataloader creation.

    Attributes:
        batch_size: Number of frames per batch.
        val_ratio: Validation split ratio.
        seed: Random seed for splitting.
        motion_files: Explicit motion NPZ paths.
        motion_file_yaml: YAML path used to discover files by group.
        motion_group: Optional group name in motion_file_yaml.
        motion_feature_keys: Feature keys concatenated into input vectors.
        motion_frame_stride: Frame stride used during loading.
        motion_normalize: Whether to normalize all frames globally.
    """

    batch_size: int = 256
    val_ratio: float = 0.2
    seed: int = 42
    motion_files: tuple[str, ...] = ()
    motion_file_yaml: str = ""
    motion_group: str = ""
    motion_feature_keys: tuple[str, ...] = ("joint_pos", "joint_vel")
    motion_frame_stride: int = 1
    motion_normalize: bool = False


def _resolve_motion_files(config: DataConfig) -> tuple[str, ...]:
    """Resolves motion files from explicit list or YAML groups.

    Args:
        config: Data configuration.

    Returns:
        Tuple of motion file paths.

    Raises:
        ValueError: If no file can be resolved.
        KeyError: If requested group does not exist.
    """
    if config.motion_files:
        return config.motion_files
    if not config.motion_file_yaml:
        raise ValueError("motion_files or motion_file_yaml must be provided.")

    grouped = collect_npz_paths(config.motion_file_yaml)
    if config.motion_group:
        if config.motion_group not in grouped:
            raise KeyError(
                f"Motion group '{config.motion_group}' not found. "
                f"Available: {list(grouped.keys())}"
            )
        return tuple(grouped[config.motion_group])

    merged: list[str] = []
    for files in grouped.values():
        merged.extend(files)
    if not merged:
        raise ValueError(f"No motion files found in yaml: {config.motion_file_yaml}")
    return tuple(merged)


def create_motion_dataloaders(config: DataConfig) -> Tuple[DataLoader, DataLoader, int]:
    """Creates frame-level train/validation loaders for motion data.

    Args:
        config: Data configuration.

    Returns:
        Tuple ``(train_loader, val_loader, input_dim)``.
    """
    motion_files = _resolve_motion_files(config)
    dataset = MotionFrameDataset(
        MotionDatasetConfig(
            motion_files=motion_files,
            feature_keys=config.motion_feature_keys,
            frame_stride=config.motion_frame_stride,
            normalize=config.motion_normalize,
        )
    )

    total = len(dataset)
    val_size = int(total * config.val_ratio)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(config.seed)
    permutation = torch.randperm(total, generator=generator).tolist()

    if val_size == 0:
        train_set = Subset(dataset, permutation)
        val_set = Subset(dataset, [])
    elif train_size == 0:
        train_set = Subset(dataset, [])
        val_set = Subset(dataset, permutation)
    else:
        train_set = Subset(dataset, permutation[:train_size])
        val_set = Subset(dataset, permutation[train_size:])

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    input_dim = int(dataset.sequences[0].shape[-1])
    return train_loader, val_loader, input_dim
