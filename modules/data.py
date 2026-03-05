"""Dataset and dataloader utilities for VAE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, MutableMapping, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from utils.load_motion_file import collect_npz_paths

from .observations import (
    ObsGroupCfg,
    ObsTermCfg,
    ObservationManager,
    ObservationsCfg,
)
from .motion_load import MotionDatasetConfig, MotionMimicDataset
from .registry import Registry


class RandomBinaryDataset(Dataset):
    """Simple synthetic dataset for quick VAE debugging and testing.

    Each sample is a binary vector generated from Bernoulli(0.5), which is
    compatible with auto/BCE reconstruction settings in VAE losses.
    """

    def __init__(
        self,
        num_samples: int,
        input_dim: int,
        seed: int = 42,
        as_image: bool = False,
        image_size: int = 28,
        image_channels: int = 1,
    ) -> None:
        """Initializes the synthetic dataset.

        Args:
            num_samples: Number of synthetic examples.
            input_dim: Flattened sample dimension.
            seed: Random seed for reproducibility.
            as_image: If True, returns ``[1, H, W]`` tensors.
            image_size: Image side length used when ``as_image=True``.
            image_channels: Number of output image channels.
        """
        generator = torch.Generator().manual_seed(seed)
        x = torch.bernoulli(
            torch.full((num_samples, input_dim), 0.5), generator=generator
        )
        if as_image:
            expected = image_channels * image_size * image_size
            if input_dim != expected:
                raise ValueError(
                    "When as_image=True, input_dim must equal "
                    "image_channels*image_size*image_size."
                )
            x = x.view(num_samples, image_channels, image_size, image_size)
        self._x = x

    def __len__(self) -> int:
        """Returns dataset size."""
        return self._x.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        """Gets one synthetic sample.

        Args:
            index: Sample index.

        Returns:
            Tensor with shape ``[input_dim]`` and float dtype.
        """
        return self._x[index].float()


class RandomSequenceDataset(Dataset):
    """Synthetic sequence dataset for future motion/time-series experiments.

    Samples are variable-length (optional) trajectories in ``R^feature_dim``.
    """

    def __init__(
        self,
        num_samples: int,
        sequence_length: int,
        feature_dim: int,
        variable_length: bool = True,
        min_sequence_length: int = 8,
        seed: int = 42,
    ) -> None:
        """Initializes synthetic sequence dataset.

        Args:
            num_samples: Number of sequence samples.
            sequence_length: Maximum sequence length.
            feature_dim: Feature dimension per timestep.
            variable_length: Whether each sample has random length.
            min_sequence_length: Lower bound for random lengths.
            seed: Random seed for reproducible synthesis.
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.variable_length = variable_length
        self.min_sequence_length = min_sequence_length
        self._generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        """Returns dataset size."""
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Builds one synthetic sequence sample.

        Args:
            index: Sample index (unused, kept for Dataset protocol).

        Returns:
            Dictionary with sequence tensor and sequence length.
        """
        del index
        if self.variable_length:
            length = int(
                torch.randint(
                    low=self.min_sequence_length,
                    high=self.sequence_length + 1,
                    size=(1,),
                    generator=self._generator,
                ).item()
            )
        else:
            length = self.sequence_length

        # Generate smooth periodic motion-like signals with random phase.
        t = torch.linspace(0.0, 1.0, steps=length)
        frequencies = (
            torch.rand((self.feature_dim,), generator=self._generator) * 3.0 + 0.5
        )
        phases = (
            torch.rand((self.feature_dim,), generator=self._generator) * 2.0 * torch.pi
        )
        sequence = torch.sin(
            t[:, None] * frequencies[None, :] * 2.0 * torch.pi + phases[None, :]
        )
        noise = (
            torch.randn((length, self.feature_dim), generator=self._generator) * 0.01
        )
        sequence = (sequence + noise).float()

        return {"sequence": sequence, "length": torch.tensor(length, dtype=torch.long)}


class DictWrapperDataset(Dataset):
    """Wraps arbitrary datasets into dictionary-based sample outputs."""

    def __init__(self, base: Dataset, input_key: str = "state") -> None:
        """Initializes wrapper.

        Args:
            base: Wrapped dataset.
            input_key: Key assigned to main input tensor.
        """
        self.base = base
        self.input_key = input_key

    def __len__(self) -> int:
        """Returns wrapped dataset size."""
        return len(self.base)

    def __getitem__(self, index: int) -> MutableMapping[str, object]:
        """Converts base sample into dictionary sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary with standardized keys.
        """
        item = self.base[index]
        if isinstance(item, dict):
            return item
        if isinstance(item, (tuple, list)):
            if len(item) == 2:
                return {self.input_key: item[0], "label": item[1]}
            return {self.input_key: item[0]}
        return {self.input_key: item}


@dataclass
class DataConfig:
    """Configuration for dataloader construction."""

    dataset: str = "random_binary"
    input_dim: int = 784
    num_samples: int = 512
    batch_size: int = 64
    val_ratio: float = 0.2
    data_root: str = "./data"
    seed: int = 42
    flatten: bool = True
    image_size: int = 28
    image_channels: int = 1
    sequence_length: int = 32
    sequence_feature_dim: int = 16
    sequence_variable_length: bool = True
    sequence_min_length: int = 8
    motion_files: tuple[str, ...] = ()
    motion_file_yaml: str = ""
    motion_group: str = ""
    motion_feature_keys: tuple[str, ...] = ("joint_pos", "joint_vel")
    motion_as_sequence: bool = True
    motion_frame_stride: int = 1
    motion_normalize: bool = False
    use_batch_protocol: bool = True
    observations: ObservationsCfg | None = None


DATASET_REGISTRY: Registry = Registry("dataset")


def create_dataloader(config: DataConfig) -> Tuple[DataLoader, DataLoader]:
    """Creates train and validation dataloaders.

    Args:
        config: DataConfig that defines dataset and loader settings.

    Returns:
        Tuple of ``(train_loader, val_loader)``.

    Raises:
        ValueError: If dataset type is unsupported.
    """
    dataset_builder = DATASET_REGISTRY.get(config.dataset)
    full_dataset = dataset_builder(config)

    if (
        isinstance(full_dataset, MotionMimicDataset)
        and not full_dataset.config.as_sequence
    ):
        train_set, val_set = _split_motion_frame_subsets(full_dataset, config)
    else:
        val_size = int(len(full_dataset) * config.val_ratio)
        train_size = len(full_dataset) - val_size
        if val_size == 0:
            train_set = full_dataset
            val_set = Subset(full_dataset, [])
        elif train_size == 0:
            train_set = Subset(full_dataset, [])
            val_set = full_dataset
        else:
            train_set, val_set = random_split(
                full_dataset,
                lengths=[train_size, val_size],
                generator=torch.Generator().manual_seed(config.seed),
            )

    collate_fn = _build_collate_fn(config)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def _split_motion_frame_subsets(
    dataset: MotionMimicDataset,
    config: DataConfig,
) -> Tuple[Subset, Subset]:
    """Splits motion dataset at frame level for frame-wise VAE sampling.

    This function explicitly uses ``sequence_lengths`` to derive the total frame
    count and create randomized frame indices. Therefore train/val partitions are
    made on frame indices instead of trajectory indices.

    Args:
        dataset: MotionMimicDataset in frame mode.
        config: Data configuration.

    Returns:
        Tuple of ``(train_subset, val_subset)`` on frame indices.
    """
    total_frames = int(sum(dataset.sequence_lengths))
    if total_frames != len(dataset):
        raise ValueError(
            "Frame-mode MotionMimicDataset size mismatch: "
            f"sum(sequence_lengths)={total_frames}, len(dataset)={len(dataset)}."
        )

    generator = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(total_frames, generator=generator).tolist()
    val_size = int(total_frames * config.val_ratio)
    train_size = total_frames - val_size

    if val_size == 0:
        return Subset(dataset, perm), Subset(dataset, [])
    if train_size == 0:
        return Subset(dataset, []), Subset(dataset, perm)
    train_indices = perm[:train_size]
    val_indices = perm[train_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


@DATASET_REGISTRY.register("random_binary")
def build_random_binary_dataset(config: DataConfig) -> Dataset:
    """Builds random binary dataset from config."""
    dataset = RandomBinaryDataset(
        num_samples=config.num_samples,
        input_dim=config.input_dim,
        seed=config.seed,
        as_image=not config.flatten,
        image_size=config.image_size,
        image_channels=config.image_channels,
    )
    return DictWrapperDataset(dataset, input_key="state")


@DATASET_REGISTRY.register("mnist")
def build_mnist_dataset(config: DataConfig) -> Dataset:
    """Builds MNIST dataset from config."""
    # Lazy import keeps the code usable even if torchvision is unavailable.
    from torchvision import datasets, transforms

    Path(config.data_root).mkdir(parents=True, exist_ok=True)
    transform_list = [transforms.ToTensor()]
    if config.flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    transform = transforms.Compose(transform_list)
    dataset = datasets.MNIST(
        root=config.data_root, train=True, download=True, transform=transform
    )
    return DictWrapperDataset(dataset, input_key="image")


@DATASET_REGISTRY.register("random_sequence")
def build_random_sequence_dataset(config: DataConfig) -> Dataset:
    """Builds random variable-length sequence dataset from config."""
    return RandomSequenceDataset(
        num_samples=config.num_samples,
        sequence_length=config.sequence_length,
        feature_dim=config.sequence_feature_dim,
        variable_length=config.sequence_variable_length,
        min_sequence_length=config.sequence_min_length,
        seed=config.seed,
    )


@DATASET_REGISTRY.register("motion_mimic")
def build_motion_mimic_dataset(config: DataConfig) -> Dataset:
    """Builds motion-mimic dataset from remapped robot motion NPZ files."""
    resolved_motion_files = config.motion_files
    if not resolved_motion_files and config.motion_file_yaml:
        grouped_files = collect_npz_paths(config.motion_file_yaml)
        if config.motion_group:
            if config.motion_group not in grouped_files:
                raise KeyError(
                    f"Motion group '{config.motion_group}' not found in yaml "
                    f"{config.motion_file_yaml}. Available: {list(grouped_files.keys())}"
                )
            resolved_motion_files = tuple(grouped_files[config.motion_group])
        else:
            merged: list[str] = []
            for files in grouped_files.values():
                merged.extend(files)
            resolved_motion_files = tuple(merged)

    motion_cfg = MotionDatasetConfig(
        motion_files=resolved_motion_files,
        feature_keys=config.motion_feature_keys,
        as_sequence=config.motion_as_sequence,
        frame_stride=config.motion_frame_stride,
        normalize=config.motion_normalize,
    )
    return MotionMimicDataset(motion_cfg)


def _build_collate_fn(config: DataConfig):
    """Builds collate function according to batch protocol config.

    Args:
        config: Dataloader configuration.

    Returns:
        Collate callable accepted by ``DataLoader``.
    """
    if not config.use_batch_protocol:
        return None

    obs_cfg = config.observations or _default_observations_cfg(config)
    manager = ObservationManager(obs_cfg)

    def _collate(samples: List[MutableMapping[str, object]]) -> Dict[str, object]:
        """Collates raw dictionary samples into protocol-compliant batch.

        Args:
            samples: Raw dataset samples.

        Returns:
            Batch dictionary with observation groups and metadata.
        """
        return manager.build_batch(samples)

    return _collate


def _default_observations_cfg(config: DataConfig) -> ObservationsCfg:
    """Creates default observation configuration by dataset type.

    Args:
        config: Data configuration.

    Returns:
        Default observation config.
    """
    dataset_name = config.dataset.lower().strip()
    if dataset_name == "random_sequence":
        policy = ObsGroupCfg(
            name="policy",
            terms=(ObsTermCfg(name="sequence", source_key="sequence"),),
            concatenate_terms=False,
        )
    elif dataset_name == "motion_mimic":
        source_key = "sequence" if config.motion_as_sequence else "state"
        policy = ObsGroupCfg(
            name="policy",
            terms=(ObsTermCfg(name=source_key, source_key=source_key),),
            concatenate_terms=False,
        )
    elif dataset_name == "mnist":
        policy = ObsGroupCfg(
            name="policy",
            terms=(ObsTermCfg(name="image", source_key="image"),),
            concatenate_terms=False,
        )
    else:
        key = "state"
        policy = ObsGroupCfg(
            name="policy",
            terms=(ObsTermCfg(name=key, source_key=key),),
            concatenate_terms=False,
        )
    return ObservationsCfg(groups=(policy,), primary_group="policy")
