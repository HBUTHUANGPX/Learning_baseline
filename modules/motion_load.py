"""Motion dataset loading utilities with temporal context indexing."""

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
        history_frames: Number of history frames in model input.
        future_frames: Number of future frames in model input.
        reconstruction_target: Target mode in ``{"all", "current", "future"}``.
        future_target_offset: Future-step offset used when target mode is
            ``"future"``.
    """

    motion_files: tuple[str, ...]
    feature_keys: tuple[str, ...] = ("joint_pos", "joint_vel")
    frame_stride: int = 1
    normalize: bool = False
    history_frames: int = 0
    future_frames: int = 0
    reconstruction_target: str = "current"
    future_target_offset: int = 1


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
    """Frame-level motion dataset with temporal context and flexible targets.

    Input frame vectors are built from a context window:
    ``[t-history, ..., t, ..., t+future]``.

    Valid centers are constrained so all required context frames exist. This
    avoids cross-boundary padding artifacts and keeps indexing deterministic.
    Indexing is always performed inside each individual sequence and never
    crosses sequence boundaries.
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
            # NOTE:
            # ``torch.cat`` is used only to compute global normalization stats.
            # It does NOT merge trajectories for indexing or sampling. The
            # training index is still built per sequence below.
            all_frames = torch.cat(self.sequences, dim=0)
            mean = all_frames.mean(dim=0, keepdim=True)
            std = all_frames.std(dim=0, keepdim=True).clamp_min(1e-6)
            self.sequences = [(sequence - mean) / std for sequence in self.sequences]

        self.sequence_lengths = [int(sequence.shape[0]) for sequence in self.sequences]
        self.frame_dim = int(self.sequences[0].shape[-1])
        self.window_size = 1 + int(config.history_frames) + int(config.future_frames)
        self.input_dim = self.window_size * self.frame_dim
        self.target_dim = self._compute_target_dim()

        # ``_center_index`` stores (sequence_id, center_frame_id). Because the
        # sequence id is explicit, each sample window is always extracted from a
        # single trajectory and cannot leak into neighboring trajectories.
        self._center_index: list[tuple[int, int]] = []
        min_center = int(config.history_frames)
        for sequence_id, sequence in enumerate(self.sequences):
            # Front boundary reserve: center >= history_frames.
            # Back boundary reserve: center <= (len(sequence)-1-future_frames).
            max_center = int(sequence.shape[0]) - 1 - int(config.future_frames)
            if max_center < min_center:
                continue
            self._center_index.extend(
                (sequence_id, frame_id)
                for frame_id in range(min_center, max_center + 1)
            )

        if not self._center_index:
            raise ValueError(
                "No valid frame centers found. Reduce history/future frames "
                "or use longer motion sequences."
            )

    def _validate_temporal_config(self) -> None:
        """Validates temporal indexing and target-mode settings."""
        if self.config.history_frames < 0 or self.config.future_frames < 0:
            raise ValueError("history_frames and future_frames must be >= 0.")
        target_mode = self.config.reconstruction_target.lower().strip()
        if target_mode not in {"all", "current", "future"}:
            raise ValueError(
                "reconstruction_target must be one of {'all', 'current', 'future'}."
            )
        if target_mode == "future":
            if self.config.future_frames <= 0:
                raise ValueError(
                    "future target requires future_frames > 0 in input context."
                )
            if not (1 <= self.config.future_target_offset <= self.config.future_frames):
                raise ValueError(
                    "future_target_offset must be within [1, future_frames]."
                )

    def _compute_target_dim(self) -> int:
        """Computes flattened target dimension from reconstruction mode."""
        mode = self.config.reconstruction_target.lower().strip()
        if mode == "all":
            return self.window_size * self.frame_dim
        return self.frame_dim

    def __len__(self) -> int:
        """Returns total number of valid center frames."""
        return len(self._center_index)

    def __getitem__(self, index: int) -> MutableMapping[str, torch.Tensor]:
        """Fetches one context-input and reconstruction-target sample.

        Args:
            index: Sample index over valid center frames.

        Returns:
            Dictionary with flattened model input and target tensors.
        """
        sequence_id, center_id = self._center_index[index]
        sequence = self.sequences[sequence_id]

        # Window slicing is performed on the selected sequence only.
        start = center_id - int(self.config.history_frames)
        end = center_id + int(self.config.future_frames) + 1
        window = sequence[start:end]  # [W, D]
        input_vector = window.reshape(-1)

        target_mode = self.config.reconstruction_target.lower().strip()
        if target_mode == "all":
            target_vector = window.reshape(-1)
        elif target_mode == "current":
            target_vector = sequence[center_id]
        else:
            target_index = center_id + int(self.config.future_target_offset)
            target_vector = sequence[target_index]

        return {
            "input": input_vector,
            "target": target_vector,
            # Backward-compatible alias for legacy debug scripts.
            "state": input_vector,
            "motion_id": torch.tensor(sequence_id, dtype=torch.long),
            "frame_id": torch.tensor(center_id, dtype=torch.long),
        }
