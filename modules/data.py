"""Data pipeline for motion VQ/FSQ training with context conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterator, MutableMapping, Tuple

import torch

from utils.load_motion_file import collect_npz_paths

from .motion_load import MotionDatasetConfig, MotionFrameDataset


class TensorIndexSubset:
    """Minimal subset view used by TensorBatchLoader."""

    def __init__(self, dataset: MotionFrameDataset, indices: torch.Tensor) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return int(self.indices.numel())


class TensorBatchLoader:
    """Batch loader with prebuilt tensor indices and vectorized dataset lookup."""

    def __init__(
        self,
        dataset: MotionFrameDataset,
        indices: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        self.dataset = TensorIndexSubset(dataset, indices.to(dtype=torch.long))
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        total = len(self.dataset)
        return int(ceil(total / self.batch_size)) if total > 0 else 0

    def __iter__(self) -> Iterator[MutableMapping[str, torch.Tensor]]:
        indices = self.dataset.indices
        if self.shuffle and indices.numel() > 1:
            generator = torch.Generator().manual_seed(self.seed + self._epoch)
            order = indices[torch.randperm(indices.numel(), generator=generator)]
        else:
            order = indices
        self._epoch += 1

        for start in range(0, int(order.numel()), self.batch_size):
            batch_indices = order[start : start + self.batch_size]
            yield self.dataset.dataset.get_batch(batch_indices)


@dataclass(frozen=True)
class DataConfig:
    """Configuration for motion dataloader creation.

    Attributes:
        batch_size: Number of samples per batch.
        val_ratio: Validation split ratio.
        seed: Random seed for splitting.
        motion_files: Explicit motion NPZ paths.
        motion_file_yaml: YAML path used to discover files by group.
        motion_group: Optional group name in motion_file_yaml.
        motion_feature_keys: Feature keys concatenated into frame vectors.
        motion_frame_stride: Frame stride used during loading.
        motion_normalize: Whether to normalize all frames globally.
        motion_cache_device: Device where motion tensors are cached.
            ``"auto"`` chooses CUDA when available, otherwise CPU.
        history_frames: Number of history frames in context/condition.
        future_frames: Number of future frames in context/target.
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
    motion_cache_device: str = "auto"
    history_frames: int = 0
    future_frames: int = 0


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


def create_motion_dataloaders(
    config: DataConfig,
) -> Tuple[TensorBatchLoader, TensorBatchLoader, int, int, int]:
    """Creates train/validation loaders for context-conditioned motion data.

    Args:
        config: Data configuration.

    Returns:
        Tuple ``(train_loader, val_loader, encoder_input_dim,
        decoder_condition_dim, target_dim)``.
    """
    motion_files = _resolve_motion_files(config)
    dataset = MotionFrameDataset(
        MotionDatasetConfig(
            motion_files=motion_files,
            feature_keys=config.motion_feature_keys,
            frame_stride=config.motion_frame_stride,
            normalize=config.motion_normalize,
            cache_device=config.motion_cache_device,
            history_frames=config.history_frames,
            future_frames=config.future_frames,
        )
    )

    total = len(dataset)
    val_size = int(total * config.val_ratio)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(config.seed)
    permutation = torch.randperm(total, generator=generator, dtype=torch.long)

    if val_size == 0:
        train_indices = permutation
        val_indices = torch.zeros(0, dtype=torch.long)
    elif train_size == 0:
        train_indices = torch.zeros(0, dtype=torch.long)
        val_indices = permutation
    else:
        train_indices = permutation[:train_size]
        val_indices = permutation[train_size:]

    train_loader = TensorBatchLoader(
        dataset=dataset,
        indices=train_indices,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_loader = TensorBatchLoader(
        dataset=dataset,
        indices=val_indices,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed + 10_000,
    )
    return (
        train_loader,
        val_loader,
        dataset.encoder_input_dim,
        dataset.decoder_condition_dim,
        dataset.target_dim,
    )
