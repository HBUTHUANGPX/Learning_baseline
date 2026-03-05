"""Minimal unit tests for training and logging flow."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import Adam

from modules.data import DataConfig, create_dataloader
from modules.vae import VanillaVAE
from scripts.train_vae import (
    ExperimentManager,
    VAETerm,
    train_one_epoch,
    validate_one_epoch,
)
from utils.tb_logger import create_experiment_paths


def test_train_one_epoch_runs_and_returns_scalars() -> None:
    """Tests one-epoch training execution on synthetic data."""
    config = DataConfig(
        dataset="random_binary", input_dim=32, num_samples=64, batch_size=16
    )
    train_loader, _ = create_dataloader(config)

    model = VanillaVAE(input_dim=32, latent_dim=4, hidden_dims=(16, 8))
    term = VAETerm(
        model=model,
        metric_keys=("loss", "recon_loss", "kl_loss"),
        expects_image_input=False,
    )
    optimizer = Adam(model.parameters(), lr=1e-3)
    metrics = train_one_epoch(term, train_loader, optimizer, device=torch.device("cpu"))

    assert "loss" in metrics and "recon_loss" in metrics and "kl_loss" in metrics
    assert metrics["loss"] > 0


def test_create_experiment_paths_creates_expected_directories(tmp_path: Path) -> None:
    """Tests whether log directories are created under timestamped run folder."""
    paths = create_experiment_paths(log_root=str(tmp_path), timestamp="20260303_000000")

    assert paths.run_dir.exists()
    assert paths.tensorboard_dir.exists()
    assert paths.checkpoint_dir.exists()
    assert paths.reconstructions_dir.exists()


def test_empty_validation_loader_is_handled(tmp_path: Path) -> None:
    """Tests empty validation boundary with manager fallback and validation API."""
    config = DataConfig(
        dataset="random_binary",
        input_dim=32,
        num_samples=8,
        batch_size=4,
        val_ratio=0.0,
    )
    train_loader, val_loader = create_dataloader(config)
    assert len(val_loader) == 0
    assert len(train_loader) > 0

    model = VanillaVAE(input_dim=32, latent_dim=4, hidden_dims=(16, 8))
    term = VAETerm(
        model=model,
        metric_keys=("loss", "recon_loss", "kl_loss"),
        expects_image_input=False,
    )
    val_metrics = validate_one_epoch(term, val_loader, device=torch.device("cpu"))
    assert set(val_metrics.keys()) == {"loss", "recon_loss", "kl_loss"}
    assert val_metrics["loss"] == 0.0

    args = argparse.Namespace(
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
        recon_loss_mode="auto",
        beta=4.0,
        num_embeddings=32,
        fsq_levels=6,
        epochs=1,
        batch_size=4,
        num_samples=8,
        sequence_length=32,
        sequence_feature_dim=16,
        sequence_variable_length=False,
        sequence_min_length=8,
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
        data_root=str(tmp_path / "data"),
        log_root=str(tmp_path / "log"),
    )
    manager = ExperimentManager(args)
    manager.train_loader = train_loader
    manager.val_loader = val_loader
    manager._save_epoch_artifacts(epoch=1)
    manager.tb_logger.close()
