"""Minimal tests for batch protocol and sequence data pipeline."""

from __future__ import annotations

from modules.data import DataConfig, create_dataloader


def test_random_binary_uses_batch_protocol() -> None:
    """Tests batch protocol output format for random binary dataset."""
    config = DataConfig(
        dataset="random_binary",
        input_dim=32,
        num_samples=32,
        batch_size=8,
        use_batch_protocol=True,
    )
    train_loader, _ = create_dataloader(config)
    batch = next(iter(train_loader))

    assert "obs" in batch
    assert "policy" in batch["obs"]
    assert batch["obs"]["policy"].shape == (8, 32)


def test_random_sequence_variable_length_produces_mask_and_lengths() -> None:
    """Tests variable-length sequence collation with mask/length metadata."""
    config = DataConfig(
        dataset="random_sequence",
        num_samples=64,
        batch_size=8,
        sequence_length=20,
        sequence_feature_dim=6,
        sequence_variable_length=True,
        sequence_min_length=5,
        use_batch_protocol=True,
    )
    train_loader, _ = create_dataloader(config)
    batch = next(iter(train_loader))

    x = batch["obs"]["policy"]
    lengths = batch["meta"]["policy_lengths"]
    mask = batch["meta"]["policy_mask"]

    assert x.ndim == 3
    assert x.shape[0] == 8
    assert x.shape[2] == 6
    assert lengths.shape == (8,)
    assert mask.shape[:2] == x.shape[:2]
