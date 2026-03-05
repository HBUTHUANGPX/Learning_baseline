"""Motion data loading utilities for VAE training pipelines.

This module adapts remapped robot motion NPZ files to the project's dataset
interface and batch protocol, independent from IsaacLab runtime classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def extract_part(path: str) -> str | None:
    """Extracts relative artifact path for NPZ files.

    Args:
        path: Input path string, optionally prefixed by ``"artifacts/"``.

    Returns:
        Relative path after ``"artifacts/"`` prefix if input is an NPZ file;
        otherwise ``None``.
    """
    if path.startswith("artifacts/"):
        relative_path = path[len("artifacts/") :]
        if relative_path.endswith(".npz"):
            return relative_path
    return None


@dataclass(frozen=True)
class MotionDatasetConfig:
    """Configuration for motion NPZ dataset loading.

    Attributes:
        motion_files: Explicit NPZ file list.
        motion_file_group: Optional grouped file definition (group -> files).
        feature_keys: NPZ keys to include in sample features.
        as_sequence: Whether one dataset item is one full sequence.
        frame_stride: Temporal stride when reading sequence frames.
        normalize: Whether to z-score normalize features with dataset statistics.
    """

    motion_files: tuple[str, ...] = ()
    motion_file_group: Mapping[str, Sequence[str]] | None = None
    feature_keys: tuple[str, ...] = ("joint_pos", "joint_vel")
    as_sequence: bool = True
    frame_stride: int = 1
    normalize: bool = False


def resolve_motion_files(
    motion_files: Sequence[str] | None,
    motion_file_group: Mapping[str, Sequence[str]] | None,
) -> list[Path]:
    """Resolves and validates NPZ file paths from list/group declarations.

    Args:
        motion_files: Direct list of NPZ files.
        motion_file_group: Grouped NPZ files.

    Returns:
        Validated file path list.

    Raises:
        ValueError: If no file is provided.
        FileNotFoundError: If any file path does not exist.
    """
    files: list[str] = []
    if motion_files:
        files.extend(motion_files)
    if motion_file_group:
        for _, group_files in motion_file_group.items():
            files.extend(list(group_files))
    if not files:
        raise ValueError("No motion files provided. Set motion_files or motion_file_group.")

    paths = [Path(file).expanduser().resolve() for file in files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Motion files not found: {missing}")
    return paths


def _flatten_feature_tensor(array: np.ndarray) -> torch.Tensor:
    """Converts NPZ feature array to ``[T, D]`` tensor.

    Args:
        array: Numpy array with first axis as time dimension.

    Returns:
        Flattened feature tensor with shape ``[T, D]``.
    """
    tensor = torch.tensor(array, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    return tensor.reshape(tensor.shape[0], -1)


def load_motion_feature_sequence(
    path: Path,
    feature_keys: Sequence[str],
    frame_stride: int = 1,
) -> torch.Tensor:
    """Loads selected NPZ features and concatenates them into one sequence.

    Args:
        path: Motion NPZ file path.
        feature_keys: NPZ keys included in output feature vector.
        frame_stride: Temporal frame stride, must be >= 1.

    Returns:
        Sequence tensor with shape ``[T, D]``.

    Raises:
        KeyError: If any required feature key is missing.
        ValueError: If feature lengths are inconsistent.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1.")

    data = np.load(str(path))
    features: list[torch.Tensor] = []
    lengths: list[int] = []
    for key in feature_keys:
        if key not in data:
            raise KeyError(f"Feature key '{key}' not found in motion file: {path}")
        tensor = _flatten_feature_tensor(data[key])
        features.append(tensor)
        lengths.append(int(tensor.shape[0]))
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent feature lengths in file {path}: {lengths}")

    sequence = torch.cat(features, dim=1)
    if frame_stride > 1:
        sequence = sequence[::frame_stride]
    return sequence


class MotionMimicDataset(Dataset):
    """Dataset for remapped robot joint motion NPZ files.

    It supports two modes:
    - sequence mode: one sample is one motion sequence ``[T, D]``
    - frame mode: one sample is one frame vector ``[D]``
    """

    def __init__(self, config: MotionDatasetConfig) -> None:
        """Initializes dataset and preloads sequences.

        Args:
            config: Motion dataset configuration.
        """
        self.config = config
        self.paths = resolve_motion_files(config.motion_files, config.motion_file_group)
        self.sequences = [
            load_motion_feature_sequence(
                path=path,
                feature_keys=config.feature_keys,
                frame_stride=config.frame_stride,
            )
            for path in self.paths
        ]
        self.sequence_lengths = [int(seq.shape[0]) for seq in self.sequences]

        self._frame_to_sequence: list[tuple[int, int]] = []
        if not config.as_sequence:
            for seq_id, sequence in enumerate(self.sequences):
                self._frame_to_sequence.extend(
                    [(seq_id, frame_id) for frame_id in range(int(sequence.shape[0]))]
                )

        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        if config.normalize:
            all_frames = torch.cat(self.sequences, dim=0)
            self.mean = all_frames.mean(dim=0, keepdim=True)
            self.std = all_frames.std(dim=0, keepdim=True).clamp_min(1e-6)
            self.sequences = [(sequence - self.mean) / self.std for sequence in self.sequences]

    def __len__(self) -> int:
        """Returns dataset size under active sampling mode."""
        if self.config.as_sequence:
            return len(self.sequences)
        return len(self._frame_to_sequence)

    def __getitem__(self, index: int) -> MutableMapping[str, object]:
        """Fetches one sample in sequence or frame mode.

        Args:
            index: Sample index.

        Returns:
            Dictionary sample compatible with observation protocol.
        """
        if self.config.as_sequence:
            sequence = self.sequences[index]
            return {
                "sequence": sequence,
                "length": torch.tensor(int(sequence.shape[0]), dtype=torch.long),
                "motion_id": torch.tensor(index, dtype=torch.long),
            }

        sequence_id, frame_id = self._frame_to_sequence[index]
        state = self.sequences[sequence_id][frame_id]
        return {
            "state": state,
            "motion_id": torch.tensor(sequence_id, dtype=torch.long),
            "frame_id": torch.tensor(frame_id, dtype=torch.long),
        }
