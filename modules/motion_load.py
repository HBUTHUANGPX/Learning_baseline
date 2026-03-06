"""Motion dataset loading utilities with context-conditioned targets."""

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
        feature_keys: NPZ keys concatenated into frame features.
        frame_stride: Temporal sampling stride.
        normalize: Whether to z-score normalize all frame features.
        history_frames: Number of history frames for encoder/decoder condition.
        future_frames: Number of future frames included in encoder input and
            target reconstruction.
    """

    motion_files: tuple[str, ...]
    feature_keys: tuple[str, ...] = ("joint_pos", "joint_vel")
    frame_stride: int = 1
    normalize: bool = False
    history_frames: int = 0
    future_frames: int = 0


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
    """Frame-level dataset with encoder context and decoder history condition.

    For each valid center frame ``t`` in one trajectory:
    - encoder input: ``[t-h, ..., t, ..., t+f]``
    - decoder condition: ``[t-h, ..., t-1]``
    - target: ``[t, ..., t+f]``

    All indexing is performed strictly inside each trajectory. Front/back ranges
    are trimmed by ``history_frames`` and ``future_frames`` to avoid crossing
    trajectory boundaries.
    """

    def __init__(self, config: MotionDatasetConfig) -> None:
        """Initializes dataset from multiple motion files.

        Args:
            config: Motion dataset configuration.
        """
        self.config = config
        self._validate_temporal_config()

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
            # ``cat`` is only used for normalization statistics.
            all_frames = torch.cat(self.sequences, dim=0)
            mean = all_frames.mean(dim=0, keepdim=True)
            std = all_frames.std(dim=0, keepdim=True).clamp_min(1e-6)
            self.sequences = [(sequence - mean) / std for sequence in self.sequences]

        self.sequence_lengths = [int(sequence.shape[0]) for sequence in self.sequences]
        self.frame_dim = int(self.sequences[0].shape[-1])

        self.encoder_window = 1 + int(config.history_frames) + int(config.future_frames)
        self.target_window = 1 + int(config.future_frames)
        self.condition_window = int(config.history_frames)

        self.encoder_input_dim = self.encoder_window * self.frame_dim
        self.decoder_condition_dim = self.condition_window * self.frame_dim
        self.target_dim = self.target_window * self.frame_dim

        self._center_index: list[tuple[int, int]] = []
        min_center = int(config.history_frames)
        for sequence_id, sequence in enumerate(self.sequences):
            max_center = int(sequence.shape[0]) - 1 - int(config.future_frames)
            if max_center < min_center:
                continue
            self._center_index.extend(
                (sequence_id, center_id)
                for center_id in range(min_center, max_center + 1)
            )

        if not self._center_index:
            raise ValueError(
                "No valid frame centers found. Reduce history/future frames "
                "or use longer motion sequences."
            )

    def _validate_temporal_config(self) -> None:
        """Validates temporal indexing settings."""
        if self.config.history_frames < 0 or self.config.future_frames < 0:
            raise ValueError("history_frames and future_frames must be >= 0.")

    def __len__(self) -> int:
        """Returns total number of valid center frames."""
        return len(self._center_index)

    def __getitem__(self, index: int) -> MutableMapping[str, torch.Tensor]:
        """Fetches one context-conditioned training sample.

        Args:
            index: Sample index over valid center frames.

        Returns:
            Dictionary with encoder input, decoder condition, and target vectors.
        """
        sequence_id, center_id = self._center_index[index]
        sequence = self.sequences[sequence_id]

        history = int(self.config.history_frames)
        future = int(self.config.future_frames)

        enc_start = center_id - history
        enc_end = center_id + future + 1
        encoder_window = sequence[enc_start:enc_end]

        cond_start = center_id - history
        cond_end = center_id
        condition_window = sequence[cond_start:cond_end]

        target_start = center_id
        target_end = center_id + future + 1
        target_window = sequence[target_start:target_end]

        encoder_input = encoder_window.reshape(-1)
        decoder_condition = condition_window.reshape(-1)
        target = target_window.reshape(-1)

        return {
            "encoder_input": encoder_input,
            "decoder_condition": decoder_condition,
            "target": target,
            # Aliases kept for simple backward compatibility.
            "input": encoder_input,
            "state": encoder_input,
            "motion_id": torch.tensor(sequence_id, dtype=torch.long),
            "frame_id": torch.tensor(center_id, dtype=torch.long),
        }
