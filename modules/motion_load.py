"""Motion dataset loading utilities for frame-level VQ/FSQ training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def _flatten_feature(array: np.ndarray) -> torch.Tensor:
    """Flattens a time-major feature array into ``[T, D]``.

    Args:
        array: Numpy array whose first axis is time.

    Returns:
        Float tensor with shape ``[T, D]``.
    """
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    return tensor.reshape(tensor.shape[0], -1)


@dataclass(frozen=True)
class MotionDatasetConfig:
    """Configuration for loading motion NPZ files.

    Attributes:
        motion_files: Explicit motion file paths.
        feature_keys: NPZ keys that are concatenated as model inputs.
        frame_stride: Temporal sampling stride.
        normalize: Whether to z-score normalize all frame features.
    """

    motion_files: tuple[str, ...]
    feature_keys: tuple[str, ...] = ("joint_pos", "joint_vel")
    frame_stride: int = 1
    normalize: bool = False


def resolve_motion_files(motion_files: Sequence[str]) -> list[Path]:
    """Resolves and validates motion NPZ paths.

    Args:
        motion_files: Input file paths.

    Returns:
        Absolute existing paths.

    Raises:
        ValueError: If no path is provided.
        FileNotFoundError: If any path does not exist.
    """
    if not motion_files:
        raise ValueError("No motion files provided.")
    paths = [Path(path).expanduser().resolve() for path in motion_files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Motion files not found: {missing}")
    return paths


def load_motion_feature_sequence(
    path: Path,
    feature_keys: Sequence[str],
    frame_stride: int = 1,
) -> torch.Tensor:
    """Loads and concatenates motion features from one NPZ file.

    Args:
        path: NPZ file path.
        feature_keys: Feature keys to concatenate.
        frame_stride: Temporal sampling stride.

    Returns:
        Tensor with shape ``[T, D]``.

    Raises:
        ValueError: If frame stride is invalid or lengths mismatch.
        KeyError: If any feature key is missing.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1.")

    with np.load(str(path)) as data:
        features: list[torch.Tensor] = []
        lengths: list[int] = []
        for key in feature_keys:
            if key not in data:
                raise KeyError(f"Feature key '{key}' not found in {path}.")
            part = _flatten_feature(data[key])
            features.append(part)
            lengths.append(int(part.shape[0]))

    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent feature lengths in {path}: {lengths}")

    sequence = torch.cat(features, dim=1)
    if frame_stride > 1:
        sequence = sequence[::frame_stride]
    return sequence


class MotionFrameDataset(Dataset):
    """Frame-level motion dataset that returns ``{"state": frame}`` samples."""

    def __init__(self, config: MotionDatasetConfig) -> None:
        """Initializes dataset from multiple motion files.

        Args:
            config: Motion dataset configuration.
        """
        self.config = config
        self.paths = resolve_motion_files(config.motion_files)
        self.sequences = [
            load_motion_feature_sequence(
                path=path,
                feature_keys=config.feature_keys,
                frame_stride=config.frame_stride,
            )
            for path in self.paths
        ]

        if config.normalize:
            all_frames = torch.cat(self.sequences, dim=0)
            mean = all_frames.mean(dim=0, keepdim=True)
            std = all_frames.std(dim=0, keepdim=True).clamp_min(1e-6)
            self.sequences = [(sequence - mean) / std for sequence in self.sequences]

        self.sequence_lengths = [int(sequence.shape[0]) for sequence in self.sequences]
        self._frame_index: list[tuple[int, int]] = []
        for sequence_id, sequence in enumerate(self.sequences):
            self._frame_index.extend(
                (sequence_id, frame_id) for frame_id in range(int(sequence.shape[0]))
            )

    def __len__(self) -> int:
        """Returns total number of frames across all sequences."""
        return len(self._frame_index)

    def __getitem__(self, index: int) -> MutableMapping[str, torch.Tensor]:
        """Fetches one frame sample.

        Args:
            index: Frame sample index.

        Returns:
            Dictionary with frame tensor and simple metadata.
        """
        sequence_id, frame_id = self._frame_index[index]
        state = self.sequences[sequence_id][frame_id]
        return {
            "state": state,
            "motion_id": torch.tensor(sequence_id, dtype=torch.long),
            "frame_id": torch.tensor(frame_id, dtype=torch.long),
        }
