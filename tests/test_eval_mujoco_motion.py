"""Minimal tests for MuJoCo eval helper logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.eval_mujoco_motion import (
    extract_qpos_from_reconstruction,
    load_eval_motion_arrays,
)


def test_extract_qpos_from_reconstruction_slice() -> None:
    """Tests qpos slicing from reconstructed feature vectors."""
    recon = np.arange(10, dtype=np.float32)
    qpos = extract_qpos_from_reconstruction(recon, qpos_dim=4, qpos_slice_start=2)
    np.testing.assert_allclose(qpos, np.array([2, 3, 4, 5], dtype=np.float32))


def test_extract_qpos_from_reconstruction_out_of_range() -> None:
    """Tests slice range validation for qpos extraction helper."""
    recon = np.arange(6, dtype=np.float32)
    with pytest.raises(ValueError):
        _ = extract_qpos_from_reconstruction(recon, qpos_dim=5, qpos_slice_start=2)


def test_load_eval_motion_arrays_shapes_and_stride(tmp_path: Path) -> None:
    """Tests motion NPZ parsing for feature concatenation and frame stride."""
    motion_path = tmp_path / "motion_eval.npz"
    np.savez(
        motion_path,
        fps=np.array(60.0, dtype=np.float32),
        joint_pos=np.random.randn(10, 6).astype(np.float32),
        joint_vel=np.random.randn(10, 6).astype(np.float32),
    )
    features, qpos, fps = load_eval_motion_arrays(
        npz_path=motion_path,
        feature_keys=("joint_pos", "joint_vel"),
        qpos_key="joint_pos",
        frame_stride=2,
    )
    assert features.shape == (5, 12)
    assert qpos.shape == (5, 6)
    assert abs(fps - 60.0) < 1e-6
