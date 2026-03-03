"""Minimal unit tests for training and logging flow."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import Adam

from modules.data import DataConfig, create_dataloader
from modules.vae import VanillaVAE
from scripts.train_vae import train_one_epoch
from utils.tb_logger import create_experiment_paths


def test_train_one_epoch_runs_and_returns_scalars() -> None:
    """Tests one-epoch training execution on synthetic data."""
    config = DataConfig(dataset="random_binary", input_dim=32, num_samples=64, batch_size=16)
    train_loader, _ = create_dataloader(config)

    model = VanillaVAE(input_dim=32, latent_dim=4, hidden_dims=(16, 8))
    optimizer = Adam(model.parameters(), lr=1e-3)
    metrics = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device=torch.device("cpu"),
        model_name="vanilla",
    )

    assert "loss" in metrics and "recon_loss" in metrics and "kl_loss" in metrics
    assert metrics["loss"] > 0


def test_create_experiment_paths_creates_expected_directories(tmp_path: Path) -> None:
    """Tests whether log directories are created under timestamped run folder."""
    paths = create_experiment_paths(log_root=str(tmp_path), timestamp="20260303_000000")

    assert paths.run_dir.exists()
    assert paths.tensorboard_dir.exists()
    assert paths.checkpoint_dir.exists()
    assert paths.reconstructions_dir.exists()
