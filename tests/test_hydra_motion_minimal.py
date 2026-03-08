"""Minimal tests for Hydra config mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.train_hydra import _cfg_to_namespace

hydra = pytest.importorskip("hydra")
compose = hydra.compose
initialize_config_dir = hydra.initialize_config_dir


def test_hydra_compose_motion_defaults() -> None:
    """Tests default Hydra composition for motion VQ/FSQ pipeline."""
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")
    args = _cfg_to_namespace(cfg)

    assert args.model in {"vq", "fsq", "ifsq"}
    assert args.batch_size > 0
    assert args.motion_file_yaml.endswith("motion_file.yaml")
