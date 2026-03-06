"""Motion dataset loading utilities with tensorized indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def _flatten_temporal_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Flattens a temporal tensor to ``[T, D]``.

    Args:
        tensor: Input tensor with time on axis-0.

    Returns:
        Flattened tensor with shape ``[T, D]``.

    Raises:
        ValueError: If tensor is scalar and has no temporal axis.
    """
    if tensor.ndim == 0:
        raise ValueError("Scalar tensor has no temporal axis and cannot be flattened to [T, D].")
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1).float()
    return tensor.reshape(tensor.shape[0], -1).float()


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
            part = _flatten_temporal_tensor(torch.as_tensor(data[key], dtype=torch.float32))
            features.append(part)
            lengths.append(int(part.shape[0]))

    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent feature lengths in {path}: {lengths}")

    sequence = torch.cat(features, dim=1)
    if frame_stride > 1:
        sequence = sequence[::frame_stride]
    return sequence


class MotionFrameDataset(Dataset):
    """Frame-level dataset with tensorized indexing and context windows.

    Data flow:
    1. Load each trajectory NPZ.
    2. Expose all NPZ keys as explicit ``self.<key>`` list members in ``__init__``.
    3. Build selected training features and concatenate them into
       ``self.sequence_bank`` with shape ``[total_frames, feature_dim]``.
    4. Build valid center indices as tensors ``self.center_seq_ids`` and
       ``self.center_local_ids`` for fast deterministic indexing.
    """

    def __init__(self, config: MotionDatasetConfig) -> None:
        """Initializes dataset from multiple motion files.

        Args:
            config: Motion dataset configuration.
        """
        self.config = config
        self._validate_temporal_config()

        self.paths = resolve_motion_files(config.motion_files)

        # Every key discovered in NPZ files is attached as ``self.<key>`` and
        # stores a per-trajectory list of raw tensors.
        self._npz_keys: list[str] = []
        self._sequence_feature_tensors: list[torch.Tensor] = []

        for path in self.paths:
            self._load_npz_to_members(path)

        if not self._sequence_feature_tensors:
            raise ValueError("No trajectory features were loaded.")

        if config.normalize:
            all_frames = torch.cat(self._sequence_feature_tensors, dim=0)
            mean = all_frames.mean(dim=0, keepdim=True)
            std = all_frames.std(dim=0, keepdim=True).clamp_min(1e-6)
            self._sequence_feature_tensors = [
                (sequence - mean) / std for sequence in self._sequence_feature_tensors
            ]

        # Tensorized sequence storage.
        sequence_lengths = torch.tensor(
            [int(sequence.shape[0]) for sequence in self._sequence_feature_tensors],
            dtype=torch.long,
        )
        self.sequence_lengths = [int(v) for v in sequence_lengths.tolist()]
        self.num_sequences = int(sequence_lengths.numel())
        self.frame_dim = int(self._sequence_feature_tensors[0].shape[-1])

        self.sequence_start = torch.zeros(self.num_sequences, dtype=torch.long)
        if self.num_sequences > 1:
            self.sequence_start[1:] = torch.cumsum(sequence_lengths[:-1], dim=0)
        self.sequence_lengths_tensor = sequence_lengths
        self.sequence_bank = torch.cat(self._sequence_feature_tensors, dim=0).contiguous()

        self.encoder_window = 1 + int(config.history_frames) + int(config.future_frames)
        self.target_window = 1 + int(config.future_frames)
        self.condition_window = int(config.history_frames)

        self.encoder_input_dim = self.encoder_window * self.frame_dim
        self.decoder_condition_dim = self.condition_window * self.frame_dim
        self.target_dim = self.target_window * self.frame_dim

        self.center_seq_ids, self.center_local_ids = self._build_center_index_tensors()
        if self.center_seq_ids.numel() == 0:
            raise ValueError(
                "No valid frame centers found. Reduce history/future frames "
                "or use longer motion sequences."
            )

    def _validate_temporal_config(self) -> None:
        """Validates temporal indexing settings."""
        if self.config.history_frames < 0 or self.config.future_frames < 0:
            raise ValueError("history_frames and future_frames must be >= 0.")

    def _load_npz_to_members(self, path: Path) -> None:
        """Loads one NPZ and writes all keys as explicit class members.

        Args:
            path: Source NPZ path.

        Raises:
            KeyError: If requested ``feature_keys`` are not found.
            ValueError: If selected feature lengths mismatch.
        """
        with np.load(str(path)) as data:
            # Explicitly expose every NPZ key as ``self.<key>`` list member.
            for key in data.files:
                if key not in self._npz_keys:
                    self._npz_keys.append(key)
                    # setattr(self, key, [])
                    setattr(self, key, torch.as_tensor(data[key], dtype=torch.float32))

            sequence_index = len(self._sequence_feature_tensors)
            parts: list[torch.Tensor] = []
            lengths: list[int] = []

            for key in self.config.feature_keys:
                if not hasattr(self, key):
                    raise KeyError(f"Feature key '{key}' not found in {path}.")
                raw_list = getattr(self, key)
                if sequence_index >= len(raw_list):
                    raise KeyError(
                        f"Feature key '{key}' is missing in {path} for sequence index {sequence_index}."
                    )
                raw_tensor = raw_list[sequence_index]
                part = _flatten_temporal_tensor(raw_tensor)
                if self.config.frame_stride > 1:
                    part = part[:: self.config.frame_stride]
                parts.append(part)
                lengths.append(int(part.shape[0]))

            if len(set(lengths)) != 1:
                raise ValueError(
                    f"Inconsistent selected feature lengths in {path}: {lengths}"
                )

            self._sequence_feature_tensors.append(torch.cat(parts, dim=1).contiguous())

    def _build_center_index_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds tensorized center index arrays.

        Returns:
            Tuple of ``(center_seq_ids, center_local_ids)`` tensors.
        """
        min_center = int(self.config.history_frames)
        center_seq_list: list[torch.Tensor] = []
        center_local_list: list[torch.Tensor] = []

        for sequence_id in range(self.num_sequences):
            seq_len = int(self.sequence_lengths_tensor[sequence_id].item())
            max_center = seq_len - 1 - int(self.config.future_frames)
            if max_center < min_center:
                continue
            centers = torch.arange(min_center, max_center + 1, dtype=torch.long)
            center_seq = torch.full_like(centers, fill_value=sequence_id)
            center_seq_list.append(center_seq)
            center_local_list.append(centers)

        if not center_seq_list:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)

        return torch.cat(center_seq_list, dim=0), torch.cat(center_local_list, dim=0)

    def __len__(self) -> int:
        """Returns total number of valid center frames."""
        return int(self.center_seq_ids.numel())

    def __getitem__(self, index: int) -> MutableMapping[str, torch.Tensor]:
        """Fetches one context-conditioned training sample.

        Args:
            index: Sample index over valid center frames.

        Returns:
            Dictionary with encoder input, decoder condition, and target vectors.
        """
        sequence_id = int(self.center_seq_ids[index].item())
        center_id = int(self.center_local_ids[index].item())
        seq_start = int(self.sequence_start[sequence_id].item())

        history = int(self.config.history_frames)
        future = int(self.config.future_frames)

        enc_start = seq_start + center_id - history
        enc_end = seq_start + center_id + future + 1
        encoder_window = self.sequence_bank[enc_start:enc_end]

        cond_start = seq_start + center_id - history
        cond_end = seq_start + center_id
        condition_window = self.sequence_bank[cond_start:cond_end]

        target_start = seq_start + center_id
        target_end = seq_start + center_id + future + 1
        target_window = self.sequence_bank[target_start:target_end]

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
