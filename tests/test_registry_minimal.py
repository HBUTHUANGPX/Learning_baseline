"""Minimal tests for model/algorithm/dataset registries."""

from __future__ import annotations

import argparse

from modules.algorithms import ALGORITHM_REGISTRY, MODEL_REGISTRY, build_algorithm_term
from modules.data import DATASET_REGISTRY, DataConfig, create_dataloader


def _build_args() -> argparse.Namespace:
    """Builds minimal CLI-like namespace for registry tests."""
    return argparse.Namespace(
        algorithm="vae",
        model="vanilla",
        dataset="random_binary",
        input_dim=32,
        image_channels=1,
        image_height=28,
        image_width=28,
        latent_dim=4,
        hidden_dims="16,8",
        conv_channels="32,64",
        conv_bottleneck_dim=256,
        vq_decoder_channels="64,32",
        activation="relu",
        beta=4.0,
        num_embeddings=32,
        fsq_levels=6,
        epochs=1,
        batch_size=4,
        num_samples=8,
        sequence_length=16,
        sequence_feature_dim=6,
        sequence_variable_length=False,
        sequence_min_length=4,
        motion_files=(),
        motion_file_yaml="",
        motion_group="",
        motion_feature_keys=("joint_pos", "joint_vel"),
        motion_as_sequence=True,
        motion_frame_stride=1,
        motion_normalize=False,
        no_batch_protocol=False,
        lr=1e-3,
        seed=42,
        deterministic=False,
        device="cpu",
        data_root="./data",
        log_root="./log",
    )


def test_registry_contains_expected_keys() -> None:
    """Tests core registries expose required built-in entries."""
    assert "vanilla" in MODEL_REGISTRY.keys()
    assert "vae" in ALGORITHM_REGISTRY.keys()
    assert "random_sequence" in DATASET_REGISTRY.keys()


def test_algorithm_term_can_be_built_from_registry() -> None:
    """Tests algorithm term construction through registry pipeline."""
    args = _build_args()
    term = build_algorithm_term(args)
    assert hasattr(term, "model")
    assert "loss" in term.metric_keys


def test_dataset_registry_path_builds_protocol_batch() -> None:
    """Tests dataset registry path returns protocol-style batches."""
    config = DataConfig(
        dataset="random_sequence",
        num_samples=16,
        batch_size=4,
        sequence_length=10,
        sequence_feature_dim=5,
        use_batch_protocol=True,
    )
    train_loader, _ = create_dataloader(config)
    batch = next(iter(train_loader))
    assert "obs" in batch and "policy" in batch["obs"]
