"""Dataset and dataloader utilities for VAE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class RandomBinaryDataset(Dataset):
    """Simple synthetic dataset for quick VAE debugging and testing.

    Each sample is a binary vector generated from Bernoulli(0.5), which is
    compatible with common BCE-based VAE reconstruction losses.
    """

    def __init__(
        self,
        num_samples: int,
        input_dim: int,
        seed: int = 42,
        as_image: bool = False,
        image_size: int = 28,
    ) -> None:
        """Initializes the synthetic dataset.

        Args:
            num_samples: Number of synthetic examples.
            input_dim: Flattened sample dimension.
            seed: Random seed for reproducibility.
            as_image: If True, returns ``[1, H, W]`` tensors.
            image_size: Image side length used when ``as_image=True``.
        """
        generator = torch.Generator().manual_seed(seed)
        x = torch.bernoulli(
            torch.full((num_samples, input_dim), 0.5), generator=generator
        )
        if as_image:
            if input_dim != image_size * image_size:
                raise ValueError("When as_image=True, input_dim must equal image_size*image_size.")
            x = x.view(num_samples, 1, image_size, image_size)
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


def create_dataloader(config: DataConfig) -> Tuple[DataLoader, DataLoader]:
    """Creates train and validation dataloaders.

    Args:
        config: DataConfig that defines dataset and loader settings.

    Returns:
        Tuple of ``(train_loader, val_loader)``.

    Raises:
        ValueError: If dataset type is unsupported.
    """
    dataset_name = config.dataset.lower().strip()

    if dataset_name == "random_binary":
        full_dataset = RandomBinaryDataset(
            num_samples=config.num_samples,
            input_dim=config.input_dim,
            seed=config.seed,
            as_image=not config.flatten,
            image_size=config.image_size,
        )
    elif dataset_name == "mnist":
        # Lazy import keeps the code usable even if torchvision is unavailable.
        from torchvision import datasets, transforms

        Path(config.data_root).mkdir(parents=True, exist_ok=True)
        transform_list = [transforms.ToTensor()]
        if config.flatten:
            transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
        transform = transforms.Compose(transform_list)
        full_dataset = datasets.MNIST(
            root=config.data_root, train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset '{config.dataset}'.")

    val_size = int(len(full_dataset) * config.val_ratio)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader
