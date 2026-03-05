"""Integration test for motion_file.yaml parsing in data loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modules.data import DataConfig, create_dataloader


def _write_motion_npz(path: Path, length: int = 10, joints: int = 6) -> None:
    """Creates one minimal motion NPZ file for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.savez(
        path,
        fps=np.array(30.0, dtype=np.float32),
        joint_pos=rng.normal(size=(length, joints)).astype(np.float32),
        joint_vel=rng.normal(size=(length, joints)).astype(np.float32),
    )


def test_motion_yaml_is_used_by_motion_mimic_builder(tmp_path: Path) -> None:
    """Tests motion_mimic dataset resolves NPZ files via YAML parser utility."""
    file_a = tmp_path / "a.npz"
    _write_motion_npz(file_a)

    yaml_path = tmp_path / "motion_file.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "motion_group:",
                "  test_group:",
                "    file_name:",
                f"      - \"{str(file_a)}\"",
                "    folder_name: []",
                "    wo_file_name: []",
                "    wo_folder_name: []",
            ]
        ),
        encoding="utf-8",
    )

    config = DataConfig(
        dataset="motion_mimic",
        batch_size=1,
        val_ratio=0.0,
        motion_files=(),
        motion_file_yaml=str(yaml_path),
        motion_group="test_group",
        motion_feature_keys=("joint_pos", "joint_vel"),
        motion_as_sequence=True,
    )
    train_loader, _ = create_dataloader(config)
    batch = next(iter(train_loader))
    assert batch["obs"]["policy"].ndim == 3
