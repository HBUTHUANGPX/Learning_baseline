"""Minimal tests for Hydra layered configuration composition."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.train_hydra import _cfg_to_namespace

hydra = pytest.importorskip("hydra")
compose = hydra.compose
initialize_config_dir = hydra.initialize_config_dir


def test_hydra_default_compose() -> None:
    """Tests default Hydra composition and namespace conversion."""
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")
    args = _cfg_to_namespace(cfg)
    assert args.algorithm == "vae"
    assert args.model in {"vanilla", "beta", "conv", "vq", "fsq"}
    assert args.dataset in {"random_binary", "mnist", "random_sequence"}


def test_hydra_override_compose() -> None:
    """Tests Hydra group override and scalar override."""
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["model=fsq", "data=mnist", "train.epochs=5", "optim.lr=0.0005"],
        )
    args = _cfg_to_namespace(cfg)
    assert args.model == "fsq"
    assert args.dataset == "mnist"
    assert args.epochs == 5
    assert abs(args.lr - 0.0005) < 1e-12
