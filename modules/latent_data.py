"""Latent-condition data pipeline for FSQ AR-LDM training.
latent_data.py
Author: HuangPeixin
Last Modified: 2026-03-09

This module bridges existing motion-frame windows with text conditions and
frozen FSQ latent targets. It is designed for the AR-LDM stage where:

1. ``cond_motion`` uses ``history_frames`` only.
2. ``cond_text`` uses one trajectory-level CLIP ``pooler_output`` token.
3. ``target_latent`` uses frozen FSQ quantized latent ``z_q``.

The module is pipeline-aware with:
- trajectory-level text token alignment via ``npz_file_name``;
- deterministic train/val split over valid center indices;
- vectorized batching compatible with pre-existing motion index tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, Iterator, Mapping, MutableMapping, Tuple

import torch
from torch import nn

from utils.load_motion_file import collect_npz_paths

from .motion_load import MotionDatasetConfig, MotionFrameDataset
from .vqvae import FrameFSQVAE


@dataclass(frozen=True)
class LatentDataConfig:
    """Configuration for latent-condition dataset and loaders.

    Args:
        batch_size: Number of samples per batch.
        val_ratio: Validation split ratio in ``[0, 1]``.
        seed: Random seed used for split and shuffle.
        motion_files: Explicit list of NPZ motion file paths.
        motion_file_yaml: YAML path used to discover grouped motion files.
        motion_group: Optional group key in ``motion_file_yaml``.
        motion_feature_keys: Feature keys that define frame-level training space.
        motion_frame_stride: Temporal stride for motion loading.
        motion_normalize: Whether to globally normalize motion features.
        motion_cache_device: Device for motion tensors.
        history_frames: Number of history frames.
        future_frames: Number of future frames.
        text_token_pt: PT path containing ``npz_file_name -> token`` mapping.
        fsq_checkpoint: Trained FSQ checkpoint path.
        fsq_device: Device for frozen FSQ encoder.
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
    history_frames: int = 2
    future_frames: int = 8
    text_token_pt: str = "tmp/clip_text_pooler_by_npz.pt"
    fsq_checkpoint: str = ""
    fsq_device: str = "auto"


def _resolve_motion_files(config: LatentDataConfig) -> tuple[str, ...]:
    """Resolves motion files from explicit list or YAML groups.

    Args:
        config: Latent data configuration object.

    Returns:
        A tuple of resolved NPZ file paths.

    Raises:
        ValueError: If no motion source is provided.
        KeyError: If requested group is missing in YAML.
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
    return tuple(merged)


def _resolve_device(raw: str) -> torch.device:
    """Resolves runtime device from string option.

    Args:
        raw: Device text in ``{"auto", "cpu", "cuda"}``.

    Returns:
        Resolved torch device.

    Raises:
        RuntimeError: If CUDA is requested but unavailable.
        ValueError: If unknown option is provided.
    """
    norm = raw.lower().strip()
    if norm == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if norm == "cpu":
        return torch.device("cpu")
    if norm == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    raise ValueError(f"Unsupported device option: {raw}")


class FrozenFSQLatentEncoder(nn.Module):
    """Frozen FSQ encoder+quantizer used to generate latent supervision targets.

    Examples:
        >>> encoder = FrozenFSQLatentEncoder("log/run/checkpoint.pt", "cpu")
        >>> x = torch.randn(4, encoder.encoder_input_dim)
        >>> z_q = encoder.encode_z_q(x)
        >>> tuple(z_q.shape)
        (4, encoder.latent_dim)
    """

    def __init__(self, checkpoint_path: str, device: str = "auto") -> None:
        """Initializes frozen FSQ latent encoder from checkpoint.

        Args:
            checkpoint_path: Path to trained FSQ checkpoint.
            device: Target device in ``{"auto", "cpu", "cuda"}``.

        Raises:
            FileNotFoundError: If checkpoint path does not exist.
            KeyError: If required fields are missing in checkpoint.
        """
        super().__init__()
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"FSQ checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu")
        for key in ("encoder_input_dim", "decoder_condition_dim", "target_dim", "model_state"):
            if key not in state:
                raise KeyError(f"Checkpoint missing required key: {key}")

        args = state.get("args", {})
        model = FrameFSQVAE(
            encoder_input_dim=int(state["encoder_input_dim"]),
            decoder_condition_dim=int(state["decoder_condition_dim"]),
            target_dim=int(state["target_dim"]),
            embedding_dim=int(args.get("embedding_dim", 32)),
            hidden_dim=int(args.get("hidden_dim", 256)),
            fsq_levels=int(args.get("fsq_levels", 8)),
            recon_loss_mode=str(args.get("recon_loss_mode", "mse")),
        )
        model.load_state_dict(state["model_state"], strict=True)
        print(model)
        model.eval().requires_grad_(False)

        self.device_runtime = _resolve_device(device)
        self.model = model.to(self.device_runtime)
        self.encoder_input_dim = int(state["encoder_input_dim"])
        self.latent_dim = int(model.embedding_dim)

    @torch.no_grad()
    def encode_z_q(self, encoder_input: torch.Tensor) -> torch.Tensor:
        """Encodes FSQ latent ``z_q`` from motion encoder input.

        Args:
            encoder_input: Motion window tensor with shape ``[B, D_enc]``.

        Returns:
            Quantized latent tensor ``[B, D_latent]`` on input device.

        Raises:
            ValueError: If input shape does not match checkpoint encoder dimension.
        """
        if encoder_input.ndim != 2:
            raise ValueError(f"encoder_input must be 2D, got {tuple(encoder_input.shape)}.")
        if encoder_input.shape[1] != self.encoder_input_dim:
            raise ValueError(
                f"encoder_input dim mismatch: got {encoder_input.shape[1]}, "
                f"expected {self.encoder_input_dim}."
            )

        src_device = encoder_input.device
        z_e = self.model.encoder(encoder_input.to(self.device_runtime))
        q = self.model.quantizer(z_e)
        return q["z_q"].to(src_device)


class LatentConditionDataset:
    """Dataset wrapper that emits AR-LDM condition tensors and latent targets.

    Note:
        This wrapper is intentionally vectorized with ``get_batch`` to align with
        current project design where index tensors are prebuilt and batched.
    """

    def __init__(
        self,
        motion_dataset: MotionFrameDataset,
        text_token_map: Mapping[str, Mapping[str, object]],
        latent_encoder: FrozenFSQLatentEncoder,
    ) -> None:
        """Initializes latent-condition dataset wrapper.

        Args:
            motion_dataset: Prebuilt motion frame dataset.
            text_token_map: Dictionary loaded from PT file keyed by ``npz_file_name``.
            latent_encoder: Frozen FSQ latent encoder used to build targets.
        """
        self.motion_dataset = motion_dataset
        self.text_token_map = text_token_map
        self.latent_encoder = latent_encoder

        self.history_frames = int(motion_dataset.history_frames)
        self.future_frames = int(motion_dataset.future_frames)
        self.frame_dim = int(motion_dataset.frame_dim)
        self.condition_frames = self.history_frames
        self.condition_motion_dim = self.condition_frames * self.frame_dim
        self.latent_dim = int(latent_encoder.latent_dim)
        if self.history_frames > (1 + self.future_frames):
            raise ValueError(
                "AR-LDM requires history_frames <= 1 + future_frames. "
                f"Got history_frames={self.history_frames}, future_frames={self.future_frames}."
            )

        self.sequence_file_names = [path.name for path in motion_dataset.paths]
        if not self.sequence_file_names:
            raise ValueError("No sequence files found in motion dataset.")

        # Infer text dimension once to validate all rows.
        sample_key = self.sequence_file_names[0]
        if sample_key not in self.text_token_map:
            raise KeyError(f"Text token map missing key: {sample_key}")
        sample_token = self._extract_text_token(self.text_token_map[sample_key])
        self.text_dim = int(sample_token.numel())
        self.text_token_bank = self._build_text_token_bank()

    def _build_text_token_bank(self) -> torch.Tensor:
        """Builds dense ``seq_id -> text_token`` bank for vectorized indexing.

        Returns:
            Text token bank with shape ``[num_sequences, text_dim]`` on the same
            device as the wrapped motion dataset cache.

        Raises:
            KeyError: If any sequence key is missing in text token map.
            ValueError: If any token dimension is inconsistent.
        """
        token_device = self.motion_dataset.sequence_bank.device
        rows: list[torch.Tensor] = []
        for npz_name in self.sequence_file_names:
            if npz_name not in self.text_token_map:
                raise KeyError(f"Text token map missing npz key: {npz_name}")
            token = self._extract_text_token(self.text_token_map[npz_name])
            if token.numel() != self.text_dim:
                raise ValueError(
                    f"Text token dim mismatch for {npz_name}: "
                    f"got {token.numel()}, expected {self.text_dim}."
                )
            rows.append(token)
        return torch.stack(rows, dim=0).to(device=token_device, dtype=torch.float32)

    def __len__(self) -> int:
        """Returns number of valid center windows in wrapped motion dataset."""
        return len(self.motion_dataset)

    def _extract_text_token(self, entry: Mapping[str, object]) -> torch.Tensor:
        """Extracts text token vector from one text-token map entry.

        Args:
            entry: One value from text token map.

        Returns:
            1D float tensor representing trajectory-level text token.

        Raises:
            KeyError: If expected token fields are missing.
            TypeError: If token field is not a tensor.
        """
        if "pooler_output" in entry:
            token = entry["pooler_output"]
        elif "token" in entry:
            token = entry["token"]
        else:
            raise KeyError("Text token entry must contain 'pooler_output' or 'token'.")
        if not isinstance(token, torch.Tensor):
            raise TypeError(f"Text token must be torch.Tensor, got {type(token)}.")
        return token.float().reshape(-1)

    def get_batch(self, batch_indices: torch.Tensor) -> MutableMapping[str, torch.Tensor]:
        """Builds one AR-LDM batch from index tensor.

        Args:
            batch_indices: 1D tensor of sample indices over valid center windows.

        Returns:
            A dictionary with:
                - ``cond_motion``: Motion condition ``[B, n*frame_dim]``.
                - ``cond_text``: Text condition token ``[B, text_dim]``.
                - ``target_latent``: Frozen FSQ quantized latent ``[B, latent_dim]``.
                - ``motion_id``: Sequence id tensor ``[B]``.
                - ``frame_id``: Center frame id tensor ``[B]``.
        """
        motion_batch = self.motion_dataset.get_batch(batch_indices)
        encoder_input = motion_batch["encoder_input"]
        motion_id = motion_batch["motion_id"]
        frame_id = motion_batch["frame_id"]

        # Condition motion uses history frames only (no current frame).
        cond_motion = encoder_input[:, : self.condition_motion_dim]

        # Vectorized token lookup by sequence id without Python loops over batch.
        cond_text = self.text_token_bank[motion_id]

        # Frozen FSQ encoder builds latent target from full encoder window.
        target_latent = self.latent_encoder.encode_z_q(encoder_input)

        return {
            "cond_motion": cond_motion,
            "cond_text": cond_text,
            "target_latent": target_latent,
            "motion_id": motion_id,
            "frame_id": frame_id,
        }


class _TensorIndexSubset:
    """Minimal subset-like container for index tensor and wrapped dataset."""

    def __init__(self, dataset: LatentConditionDataset, indices: torch.Tensor) -> None:
        self.dataset = dataset
        index_device = dataset.motion_dataset.center_seq_ids.device
        self.indices = indices.to(device=index_device, dtype=torch.long)

    def __len__(self) -> int:
        """Returns subset sample count."""
        return int(self.indices.numel())


class LatentConditionBatchLoader:
    """Batch loader for ``LatentConditionDataset`` with prebuilt index tensors.

    Examples:
        >>> for batch in loader:
        ...     print(batch["cond_motion"].shape, batch["target_latent"].shape)
    """

    def __init__(
        self,
        dataset: LatentConditionDataset,
        indices: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        """Initializes latent-condition batch loader.

        Args:
            dataset: Wrapped latent-condition dataset.
            indices: Sample index tensor.
            batch_size: Batch size.
            shuffle: Whether to shuffle each epoch.
            seed: Base random seed for shuffle.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        self.dataset = _TensorIndexSubset(dataset, indices)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        """Returns number of batches per epoch."""
        total = len(self.dataset)
        return int(ceil(total / self.batch_size)) if total > 0 else 0

    def __iter__(self) -> Iterator[MutableMapping[str, torch.Tensor]]:
        """Yields latent-condition batches for one epoch."""
        indices = self.dataset.indices
        if self.shuffle and indices.numel() > 1:
            generator = torch.Generator().manual_seed(self.seed + self._epoch)
            # generator = torch.Generator(device=indices.device).manual_seed(self.seed + self._epoch)
            # Build permutation on CPU for deterministic generator behavior, then
            # move to index tensor device for advanced indexing.
            order_idx = torch.randperm(indices.numel(), generator=generator)
            if indices.device.type != "cpu":
                order_idx = order_idx.to(indices.device)
            order = indices[order_idx]
        else:
            order = indices
        self._epoch += 1
        for start in range(0, int(order.numel()), self.batch_size):
            batch_indices = order[start : start + self.batch_size]
            yield self.dataset.dataset.get_batch(batch_indices)


def create_latent_condition_loaders(
    config: LatentDataConfig,
) -> Tuple[LatentConditionBatchLoader, LatentConditionBatchLoader, Dict[str, int]]:
    """Builds train/val loaders for AR-LDM latent diffusion training.

    Args:
        config: Latent data configuration.

    Returns:
        Tuple ``(train_loader, val_loader, meta)`` where ``meta`` contains:
            - ``cond_motion_dim``
            - ``text_dim``
            - ``latent_dim``
            - ``history_frames``
            - ``future_frames``
    """
    motion_files = _resolve_motion_files(config)
    motion_dataset = MotionFrameDataset(
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

    text_pt_path = Path(config.text_token_pt).expanduser().resolve()
    if not text_pt_path.is_file():
        raise FileNotFoundError(f"text token pt not found: {text_pt_path}")
    text_token_map = torch.load(text_pt_path, map_location="cpu")
    if not isinstance(text_token_map, dict):
        raise TypeError("text token pt must store a dict.")

    latent_encoder = FrozenFSQLatentEncoder(
        checkpoint_path=config.fsq_checkpoint,
        device=config.fsq_device,
    )
    dataset = LatentConditionDataset(
        motion_dataset=motion_dataset,
        text_token_map=text_token_map,
        latent_encoder=latent_encoder,
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

    train_loader = LatentConditionBatchLoader(
        dataset=dataset,
        indices=train_indices,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_loader = LatentConditionBatchLoader(
        dataset=dataset,
        indices=val_indices,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed + 10_000,
    )

    meta = {
        "cond_motion_dim": dataset.condition_motion_dim,
        "text_dim": dataset.text_dim,
        "latent_dim": dataset.latent_dim,
        "history_frames": dataset.history_frames,
        "future_frames": dataset.future_frames,
    }
    return train_loader, val_loader, meta
