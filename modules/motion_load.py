"""Motion dataset loading utilities with tensorized indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from utils.urdf_graph import UrdfGraph
from utils.math import *


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
        raise ValueError(
            "Scalar tensor has no temporal axis and cannot be flattened to [T, D]."
        )
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
        cache_device: Device where dataset tensors are cached.
            ``"auto"`` chooses CUDA when available, otherwise CPU.
        history_frames: Number of history frames for encoder/decoder condition.
        future_frames: Number of future frames included in encoder input and
            target reconstruction.
    """

    motion_files: tuple[str, ...]
    feature_keys: tuple[str, ...] = ("joint_pos", "joint_vel")
    frame_stride: int = 1
    normalize: bool = False
    cache_device: str = "auto"
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
            part = _flatten_temporal_tensor(
                torch.as_tensor(data[key], dtype=torch.float32)
            )
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
    3. (Optional) Prepare extra feature members derived from raw NPZ members.
    4. Build selected training features and concatenate them into
       ``self.sequence_bank`` with shape ``[total_frames, feature_dim]``.
    5. Build valid center indices as tensors ``self.center_seq_ids`` and
       ``self.center_local_ids`` for fast deterministic indexing.
    """

    def __init__(self, config: MotionDatasetConfig) -> None:
        """Initializes dataset from multiple motion files.

        Args:
            config: Motion dataset configuration.
        """
        self.config = config
        self._validate_temporal_config()
        self.cache_device = self._resolve_cache_device(config.cache_device)

        self.paths = resolve_motion_files(config.motion_files)

        # NPZ keys are first loaded as ``list[Tensor]`` then finalized to Tensor.
        self._npz_keys: list[str] = []
        self._member_sequence_lengths: dict[str, list[int]] = {}
        self._member_sequence_start: dict[str, torch.Tensor] = {}
        self._sequence_feature_tensors: list[torch.Tensor] = []

        self._prepare_feature_members()
        for path in self.paths:
            self._read_npz_to_member_lists(path)

        self._finalize_npz_member_cache()
        self._sequence_feature_tensors = self._build_sequence_feature_tensors()

        if not self._sequence_feature_tensors:
            raise ValueError("No trajectory feature tensors were built.")

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
            device=self.cache_device,
        )
        self.sequence_lengths = [int(v) for v in sequence_lengths.tolist()]
        self.num_sequences = int(sequence_lengths.numel())
        self.frame_dim = int(self._sequence_feature_tensors[0].shape[-1])

        self.sequence_start = torch.zeros(
            self.num_sequences, dtype=torch.long, device=self.cache_device
        )
        if self.num_sequences > 1:
            self.sequence_start[1:] = torch.cumsum(sequence_lengths[:-1], dim=0)
        self.sequence_lengths_tensor = sequence_lengths
        self.sequence_bank = torch.cat(
            self._sequence_feature_tensors, dim=0
        ).contiguous()

        self.encoder_window = 1 + int(config.history_frames) + int(config.future_frames)
        self.target_window = 1 + int(config.future_frames)
        self.condition_window = int(config.history_frames)
        self.history_frames = int(config.history_frames)
        self.future_frames = int(config.future_frames)

        self.encoder_input_dim = self.encoder_window * self.frame_dim
        self.decoder_condition_dim = self.condition_window * self.frame_dim
        self.target_dim = self.target_window * self.frame_dim

        self.center_seq_ids, self.center_local_ids = self._build_center_index_tensors()
        if self.center_seq_ids.numel() == 0:
            raise ValueError(
                "No valid frame centers found. Reduce history/future frames "
                "or use longer motion sequences."
            )
        self._encoder_offsets = torch.arange(
            -self.history_frames,
            self.future_frames + 1,
            dtype=torch.long,
            device=self.sequence_bank.device,
        )
        self._condition_offsets = torch.arange(
            -self.history_frames,
            0,
            dtype=torch.long,
            device=self.sequence_bank.device,
        )
        self._target_offsets = torch.arange(
            0,
            self.future_frames + 1,
            dtype=torch.long,
            device=self.sequence_bank.device,
        )

    def _validate_temporal_config(self) -> None:
        """Validates temporal indexing settings."""
        if self.config.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1.")
        if self.config.history_frames < 0 or self.config.future_frames < 0:
            raise ValueError("history_frames and future_frames must be >= 0.")

    def _resolve_cache_device(self, raw_device: str) -> torch.device:
        """Resolves tensor cache device from config."""
        if raw_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if raw_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "cache_device='cuda' was requested but CUDA is unavailable."
                )
            return torch.device("cuda")
        if raw_device == "cpu":
            return torch.device("cpu")
        raise ValueError("cache_device must be one of: auto, cpu, cuda.")

    def _read_npz_to_member_lists(self, path: Path) -> None:
        """Loads one NPZ and writes all keys as explicit class members.

        Args:
            path: Source NPZ path.
        """
        with np.load(str(path)) as data:
            # Explicitly expose every NPZ key as ``self.<key>`` list member.
            for key in data.files:
                if key not in self._npz_keys:
                    self._npz_keys.append(key)
                    setattr(self, key, [])
                    if key == "body_quat_w":
                        self._npz_keys.append("root_quat_w")
                        setattr(self, "root_quat_w", [])
                        self._npz_keys.append("root_euler_w")
                        setattr(self, "root_euler_w", [])
                        self._npz_keys.append("continuous_trigonometric_encoding")
                        setattr(self, "continuous_trigonometric_encoding", [])
                        self._npz_keys.append("delta_yaw")
                        setattr(self, "delta_yaw", [])
                    if key == "joint_pos":
                        self._npz_keys.append("delta_joint_pos")
                        setattr(self, "delta_joint_pos", [])
                    if key == "body_pos_w":
                        self._npz_keys.append("root_pos_w")
                        setattr(self, "root_pos_w", [])
                        self._npz_keys.append("root_height")
                        setattr(self, "root_height", [])
                        self._npz_keys.append("root_xy_pos")
                        setattr(self, "root_xy_pos", [])

                _tensor = torch.as_tensor(data[key], dtype=torch.float32)
                getattr(self, key).append(_tensor)
                if key == "body_quat_w":
                    root_quat_w = _tensor[
                        :, self.motion_reference_body_names_in_isaacsim_index, :
                    ]
                    root_euler_w = torch.stack(euler_xyz_from_quat(root_quat_w), dim=-1)
                    continuous_trigonometric_encoding = torch.cat(
                        [
                            torch.sin(root_euler_w[:, 0:1]),
                            torch.cos(root_euler_w[:, 0:1]) - 1,
                            torch.sin(root_euler_w[:, 1:2]),
                            torch.cos(root_euler_w[:, 1:2]) - 1,
                        ],
                        dim=1,
                    )
                    # ∆ψt = yawt+1 − yawt denotes the incremental change of the root yaw
                    diffs = torch.diff(
                        root_euler_w[:, 2:3], dim=0
                    )  # 相邻差分，形状 (N-1, 1)
                    delta_yaw = torch.cat([diffs[0:1], diffs], dim=0)
                    getattr(self, "root_quat_w").append(root_quat_w)
                    getattr(self, "root_euler_w").append(root_euler_w)
                    getattr(self, "continuous_trigonometric_encoding").append(
                        continuous_trigonometric_encoding
                    )
                    getattr(self, "delta_yaw").append(delta_yaw)
                if key == "joint_pos":
                    diffs = torch.diff(_tensor, dim=0)
                    delta_joint_pos = torch.cat([diffs[0:1], diffs], dim=0)
                    getattr(self, "delta_joint_pos").append(delta_joint_pos)
                if key == "body_pos_w":
                    root_pos_w = _tensor[
                        :, self.motion_reference_body_names_in_isaacsim_index, :
                    ]
                    root_height = root_pos_w[:, 2:3]
                    root_xy_pos = root_pos_w[:, 0:2] - root_pos_w[0:1, 0:2]
                    getattr(self, "root_pos_w").append(root_pos_w)
                    getattr(self, "root_height").append(root_height)
                    getattr(self, "root_xy_pos").append(root_xy_pos)

    def _finalize_npz_member_cache(self) -> None:
        """Finalizes NPZ member lists into tensor members."""
        for key in self._npz_keys:
            sequence_tensors: list[torch.Tensor] = getattr(self, key)
            if not sequence_tensors:
                continue

            if all(tensor.ndim == 0 for tensor in sequence_tensors):
                setattr(self, key, torch.stack(sequence_tensors, dim=0).contiguous())
                lengths = [1] * len(sequence_tensors)
                self._member_sequence_lengths[key] = lengths
                self._member_sequence_start[key] = self._build_sequence_start_tensor(
                    lengths
                )
                continue

            base_shape = sequence_tensors[0].shape[1:]
            can_cat = all(
                tensor.ndim >= 1 and tensor.shape[1:] == base_shape
                for tensor in sequence_tensors
            )
            if not can_cat:
                raise ValueError(
                    f"Inconsistent NPZ key '{key}' shapes across files: "
                    f"{[tuple(t.shape) for t in sequence_tensors]}"
                )

            lengths = [int(tensor.shape[0]) for tensor in sequence_tensors]

            setattr(
                self,
                key,
                torch.cat(sequence_tensors, dim=0).to(self.cache_device).contiguous(),
            )
            self._member_sequence_lengths[key] = lengths
            self._member_sequence_start[key] = self._build_sequence_start_tensor(
                lengths
            )

    def _prepare_feature_members(self) -> None:
        """Hook for user-defined member preprocessing before feature building.

        ``config.feature_keys`` can reference either:
        1. raw NPZ keys already exposed as ``self.<key>``, or
        2. new members built here.
        """
        # Reserved for custom feature construction from raw members.
        # Example:
        # self.my_feature = [build(seq) for seq in self.some_raw_key]
        urdf_graph = UrdfGraph(
            "/home/hpx/HPX_LOCO_2/mimic_baseline/general_motion_tracker_whole_body_teleoperation/general_motion_tracker_whole_body_teleoperation/assets/Q1/urdf/Q1_wo_hand_rl.urdf"
        )  # Example of using UrdfGraph if needed.
        isaac_sim_link_name = urdf_graph.bfs_link_order()
        motion_reference_body = "torso_link"
        motion_body_names = [
            "pelvis_link",
            "L_hip_yaw_link",
            "L_knee_link",
            "L_ankle_roll_link",
            "R_hip_yaw_link",
            "R_knee_link",
            "R_ankle_roll_link",
            "torso_link",
            "L_shoulder_roll_link",
            "L_elbow_link",
            "L_wrist_pitch_link",
            "R_shoulder_roll_link",
            "R_elbow_link",
            "R_wrist_pitch_link",
            "head_pitch_link",
        ]
        self.motion_body_names_in_isaacsim_index = [
            isaac_sim_link_name.index(name) for name in motion_body_names
        ]
        self.motion_reference_body_names_in_isaacsim_index = motion_body_names.index(
            motion_reference_body
        )
        return

    def _build_sequence_feature_tensors(self) -> list[torch.Tensor]:
        """Builds per-sequence frame features from configured member keys."""
        sequence_features: list[torch.Tensor] = []
        num_sequences = len(self.paths)

        for sequence_id in range(num_sequences):
            parts: list[torch.Tensor] = []
            lengths: list[int] = []

            for key in self.config.feature_keys:
                member_tensor = self._get_member_sequence_tensor(key, sequence_id)
                part = _flatten_temporal_tensor(member_tensor)
                if self.config.frame_stride > 1:
                    part = part[:: self.config.frame_stride]
                parts.append(part)
                lengths.append(int(part.shape[0]))

            if len(set(lengths)) != 1:
                raise ValueError(
                    f"Inconsistent selected feature lengths in sequence {sequence_id}: {lengths}"
                )
            sequence_features.append(torch.cat(parts, dim=1).contiguous())

        return sequence_features

    def _get_member_sequence_tensor(self, key: str, sequence_id: int) -> torch.Tensor:
        """Returns one sequence tensor from a tensorized member."""
        if not hasattr(self, key):
            raise KeyError(
                f"Feature key '{key}' is not available as dataset member. "
                "If this is a derived feature, create self.<key> in "
                "_prepare_feature_members()."
            )
        self._ensure_member_tensorized(key)
        lengths = self._member_sequence_lengths.get(key)
        starts = self._member_sequence_start.get(key)
        if lengths is None or starts is None:
            raise KeyError(
                f"Feature key '{key}' is missing sequence slicing metadata. "
                "For derived members, set _member_sequence_lengths and _member_sequence_start "
                "in _prepare_feature_members()."
            )
        if sequence_id >= len(lengths):
            raise IndexError(
                f"Feature member '{key}' has {len(lengths)} sequences, "
                f"but sequence index {sequence_id} was requested."
            )
        seq_start = int(starts[sequence_id].item())
        seq_end = seq_start + int(lengths[sequence_id])
        return getattr(self, key)[seq_start:seq_end]

    def _ensure_member_tensorized(self, key: str) -> None:
        """Converts list[Tensor] member to a concatenated tensor in-place."""
        member = getattr(self, key)
        if isinstance(member, torch.Tensor):
            return
        if not isinstance(member, list):
            raise TypeError(
                f"Dataset member '{key}' must be Tensor or list[Tensor], got {type(member)}."
            )
        if not member:
            raise ValueError(f"Dataset member '{key}' is empty.")

        if all(tensor.ndim == 0 for tensor in member):
            lengths = [1] * len(member)
            tensorized = torch.stack(member, dim=0).contiguous()
        else:
            base_shape = member[0].shape[1:]
            can_cat = all(
                tensor.ndim >= 1 and tensor.shape[1:] == base_shape for tensor in member
            )
            if not can_cat:
                raise ValueError(
                    f"Cannot concatenate feature member '{key}' with shapes: "
                    f"{[tuple(t.shape) for t in member]}"
                )
            lengths = [int(tensor.shape[0]) for tensor in member]
            tensorized = torch.cat(member, dim=0).contiguous()

        setattr(self, key, tensorized)
        self._member_sequence_lengths[key] = lengths
        self._member_sequence_start[key] = self._build_sequence_start_tensor(lengths)

    def _build_sequence_start_tensor(self, lengths: list[int]) -> torch.Tensor:
        """Builds per-sequence start offsets from sequence lengths."""
        start = torch.zeros(len(lengths), dtype=torch.long)
        if len(lengths) > 1:
            start[1:] = torch.cumsum(
                torch.tensor(lengths[:-1], dtype=torch.long), dim=0
            )
        return start

    def _build_center_index_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds tensorized center index arrays.

        Returns:
            Tuple of ``(center_seq_ids, center_local_ids)`` tensors.
        """
        min_center = self.history_frames
        center_seq_list: list[torch.Tensor] = []
        center_local_list: list[torch.Tensor] = []

        for sequence_id in range(self.num_sequences):
            seq_len = int(self.sequence_lengths_tensor[sequence_id].item())
            max_center = seq_len - 1 - self.future_frames
            if max_center < min_center:
                continue
            centers = torch.arange(min_center, max_center + 1, dtype=torch.long)
            center_seq = torch.full_like(centers, fill_value=sequence_id)
            center_seq_list.append(center_seq)
            center_local_list.append(centers)

        if not center_seq_list:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)

        return torch.cat(center_seq_list, dim=0).to(self.cache_device), torch.cat(
            center_local_list, dim=0
        ).to(self.cache_device)

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
        batch = self.get_batch(
            torch.tensor([index], dtype=torch.long, device=self.center_seq_ids.device)
        )
        return {
            "encoder_input": batch["encoder_input"][0],
            "decoder_condition": batch["decoder_condition"][0],
            "target": batch["target"][0],
            "input": batch["input"][0],
            "state": batch["state"][0],
            "motion_id": batch["motion_id"][0],
            "frame_id": batch["frame_id"][0],
        }

    def get_batch(
        self, batch_indices: torch.Tensor
    ) -> MutableMapping[str, torch.Tensor]:
        """Builds one batch with tensorized window indexing.

        Args:
            batch_indices: 1D tensor of sample indices over valid center frames.

        Returns:
            Dictionary with batched encoder input, decoder condition, and target.
        """
        if batch_indices.ndim != 1:
            raise ValueError("batch_indices must be a 1D tensor.")
        sample_ids = batch_indices.to(
            device=self.center_seq_ids.device, dtype=torch.long
        )
        sequence_ids = self.center_seq_ids[sample_ids]
        center_ids = self.center_local_ids[sample_ids]
        base_positions = self.sequence_start[sequence_ids] + center_ids

        encoder_index = base_positions[:, None] + self._encoder_offsets[None, :]
        encoder_window = self.sequence_bank[encoder_index]
        encoder_input = encoder_window.reshape(encoder_window.shape[0], -1)

        if self.condition_window > 0:
            condition_index = base_positions[:, None] + self._condition_offsets[None, :]
            condition_window = self.sequence_bank[condition_index]
            decoder_condition = condition_window.reshape(condition_window.shape[0], -1)
        else:
            decoder_condition = self.sequence_bank.new_zeros(
                (encoder_input.shape[0], 0)
            )

        target_index = base_positions[:, None] + self._target_offsets[None, :]
        target_window = self.sequence_bank[target_index]
        target = target_window.reshape(target_window.shape[0], -1)

        return {
            "encoder_input": encoder_input,
            "decoder_condition": decoder_condition,
            "target": target,
            "input": encoder_input,
            "state": encoder_input,
            "motion_id": sequence_ids,
            "frame_id": center_ids,
        }
