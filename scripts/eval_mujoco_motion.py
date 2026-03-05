"""Evaluates trained VAE-family checkpoints with MuJoCo sequential playback.

This script loads one motion file, runs frame-wise inference with a trained
checkpoint, and writes inferred qpos values into ``MjData.qpos`` for playback.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modules.algorithms import build_algorithm_term
from modules.motion_load import resolve_motion_files
from utils.load_motion_file import collect_npz_paths

from utils.urdf_graph import UrdfGraph

def _to_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    """Converts comma-separated text or iterables to a cleaned tuple.

    Args:
        value: Input value from CLI or checkpoint args.

    Returns:
        Tuple containing non-empty string items.
    """
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def _flatten_time_feature(array: np.ndarray) -> np.ndarray:
    """Flattens one motion feature array to ``[T, D]``.

    Args:
        array: Input array with time on axis 0.

    Returns:
        Flattened float32 array with shape ``[T, D]``.
    """
    if array.ndim == 1:
        return array.astype(np.float32)[:, None]
    return array.astype(np.float32).reshape(array.shape[0], -1)


def load_eval_motion_arrays(
    npz_path: Path,
    feature_keys: Sequence[str],
    qpos_key: str,
    frame_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Loads inference inputs and ground-truth qpos arrays from one NPZ file.

    Args:
        npz_path: Motion file path.
        feature_keys: Feature keys used as model input.
        qpos_key: Key used to extract qpos target.
        frame_stride: Temporal sampling stride.

    Returns:
        Tuple of ``(features, qpos, fps)`` where:
        - ``features`` shape is ``[T, D_in]``
        - ``qpos`` shape is ``[T, D_qpos]``
        - ``fps`` is playback frames per second

    Raises:
        KeyError: If required keys are missing.
        ValueError: If frame lengths are inconsistent.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1.")
    with np.load(str(npz_path)) as data:
        parts = []
        lengths = []
        for key in feature_keys:
            if key not in data:
                raise KeyError(f"Feature key '{key}' not found in {npz_path}.")
            part = _flatten_time_feature(data[key])
            parts.append(part)
            lengths.append(int(part.shape[0]))
        if qpos_key not in data:
            raise KeyError(f"qpos_key '{qpos_key}' not found in {npz_path}.")
        qpos = _flatten_time_feature(data[qpos_key])
        lengths.append(int(qpos.shape[0]))

        if len(set(lengths)) != 1:
            raise ValueError(f"Inconsistent temporal lengths in {npz_path}: {lengths}")

        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 30.0
    features = np.concatenate(parts, axis=1)
    if frame_stride > 1:
        features = features[::frame_stride]
        qpos = qpos[::frame_stride]
    return features, qpos, fps


def extract_qpos_from_reconstruction(
    recon: np.ndarray,
    qpos_dim: int,
    qpos_slice_start: int = 0,
) -> np.ndarray:
    """Extracts qpos slice from reconstructed feature vector.

    Args:
        recon: Reconstructed feature vector with shape ``[D_in]``.
        qpos_dim: Number of qpos values to write.
        qpos_slice_start: Start index in reconstructed vector.

    Returns:
        Qpos vector with shape ``[qpos_dim]``.

    Raises:
        ValueError: If slicing range exceeds reconstruction dimension.
    """
    end = qpos_slice_start + qpos_dim
    if recon.ndim != 1:
        raise ValueError(f"Expected 1D reconstruction vector, got shape {recon.shape}.")
    if qpos_slice_start < 0 or end > recon.shape[0]:
        raise ValueError(
            "Requested qpos slice is out of range: "
            f"start={qpos_slice_start}, dim={qpos_dim}, recon_dim={recon.shape[0]}."
        )
    return recon[qpos_slice_start:end]


def _resolve_motion_path(
    motion_file: str,
    motion_file_yaml: str,
    motion_group: str,
) -> Path:
    """Resolves one motion file path from direct path or yaml group settings.

    Args:
        motion_file: Optional direct motion file.
        motion_file_yaml: Optional yaml file that defines motion groups.
        motion_group: Group name used with motion yaml.

    Returns:
        Resolved NPZ path.

    Raises:
        ValueError: If no valid source is provided.
        KeyError: If yaml group does not exist.
    """
    if motion_file:
        return Path(motion_file).expanduser().resolve()
    if not motion_file_yaml:
        raise ValueError("Provide --motion-file or --motion-file-yaml.")
    groups = collect_npz_paths(motion_file_yaml)
    if not groups:
        raise ValueError(f"No motion files found in yaml: {motion_file_yaml}")
    if motion_group:
        if motion_group not in groups:
            raise KeyError(
                f"Motion group '{motion_group}' not found. Available: {list(groups.keys())}"
            )
        files = groups[motion_group]
    else:
        first_group = next(iter(groups.keys()))
        files = groups[first_group]
    paths = resolve_motion_files(motion_files=files, motion_file_group=None)
    return paths[0]


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for MuJoCo evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate VAE checkpoint in MuJoCo.")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pt path.")
    parser.add_argument("--mjcf", type=str, default="/home/hpx/HPX_LOCO_2/mimic_baseline/general_motion_tracker_whole_body_teleoperation/general_motion_tracker_whole_body_teleoperation/assets/Q1/mjcf/Q1_wo_hand.xml", help="MuJoCo XML/MJCF path.")
    parser.add_argument("--urdf", type=str, default="/home/hpx/HPX_LOCO_2/mimic_baseline/general_motion_tracker_whole_body_teleoperation/general_motion_tracker_whole_body_teleoperation/assets/Q1/urdf/Q1_wo_hand_rl.urdf", help="URDF path.")
    parser.add_argument("--motion-file", type=str, default="", help="Motion NPZ path.")
    parser.add_argument(
        "--motion-file-yaml",
        type=str,
        default="",
        help="Motion yaml path used when --motion-file is empty.",
    )
    parser.add_argument(
        "--motion-group",
        type=str,
        default="",
        help="Motion group key in motion yaml.",
    )
    parser.add_argument(
        "--motion-feature-keys",
        type=str,
        default="joint_pos,joint_vel",
        help="Comma-separated feature keys for model input.",
    )
    parser.add_argument(
        "--qpos-key",
        type=str,
        default="joint_pos",
        help="NPZ key used as qpos reference.",
    )
    parser.add_argument(
        "--qpos-slice-start",
        type=int,
        default=0,
        help="Start index in reconstructed vector for qpos extraction.",
    )
    parser.add_argument(
        "--qpos-start-idx",
        type=int,
        default=0,
        help="Start index in MuJoCo data.qpos to write inferred qpos.",
    )
    parser.add_argument(
        "--qpos-dim",
        type=int,
        default=0,
        help="qpos dimension to write; 0 means auto from qpos-key.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Frame stride for motion sampling.",
    )
    parser.add_argument(
        "--write-source",
        type=str,
        choices=["recon", "gt"],
        default="recon",
        help="Write reconstructed qpos or ground-truth qpos to MuJoCo.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer. Still performs sequential inference and qpos writing.",
    )
    parser.add_argument(
        "--save-recon-qpos",
        type=str,
        default="",
        help="Optional output .npz path to save reconstructed qpos sequence.",
    )
    return parser.parse_args()


def main() -> None:
    """Runs frame-wise evaluation and MuJoCo playback."""
    args = parse_args()
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    if "args" not in checkpoint:
        raise KeyError("Checkpoint missing 'args'. Expected training args in checkpoint.")
    ckpt_args = dict(checkpoint["args"])

    # Build a runtime namespace compatible with model registry builders.
    runtime_args = SimpleNamespace(**ckpt_args)
    runtime_args.device = args.device
    runtime_args.dataset = "motion_mimic"
    runtime_args.motion_as_sequence = False
    runtime_args.motion_frame_stride = int(args.frame_stride)
    runtime_args.motion_feature_keys = _to_tuple(args.motion_feature_keys)

    feature_keys = _to_tuple(args.motion_feature_keys)
    motion_path = _resolve_motion_path(
        motion_file=args.motion_file,
        motion_file_yaml=args.motion_file_yaml,
        motion_group=args.motion_group,
    )
    features, qpos_gt, fps = load_eval_motion_arrays(
        npz_path=motion_path,
        feature_keys=feature_keys,
        qpos_key=args.qpos_key,
        frame_stride=args.frame_stride,
    )

    runtime_args.input_dim = int(features.shape[1])
    runtime_args.sequence_feature_dim = int(features.shape[1])
    term = build_algorithm_term(runtime_args)
    model = term.model
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    qpos_dim = int(args.qpos_dim) if int(args.qpos_dim) > 0 else int(qpos_gt.shape[1])
    if args.qpos_start_idx < 0:
        raise ValueError("--qpos-start-idx must be >= 0.")

    import mujoco  # pylint: disable=import-outside-toplevel
    from mujoco import viewer  # pylint: disable=import-outside-toplevel

    mjcf_path = Path(args.mjcf).expanduser().resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")
    mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    mj_data = mujoco.MjData(mj_model)
    urdf_path = Path(args.urdf).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    urdf_graph = UrdfGraph(urdf_path)
    isaac_sim_joint_name = urdf_graph.bfs_joint_order()
    mujoco_joint_name = urdf_graph.joint_order_by_file()
    print(f"Resolved {len(isaac_sim_joint_name)} joints from URDF for potential qpos mapping.")
    isaac_sim2mujoco_index = [
        isaac_sim_joint_name.index(name) for name in mujoco_joint_name
    ]
    
    if args.qpos_start_idx + qpos_dim > mj_data.qpos.shape[0]:
        raise ValueError(
            "qpos write range exceeds MuJoCo qpos size: "
            f"start={args.qpos_start_idx}, dim={qpos_dim}, nq={mj_data.qpos.shape[0]}."
        )

    reconstructed_qpos = np.zeros((features.shape[0], qpos_dim), dtype=np.float32)
    frame_dt = 1.0 / max(float(fps), 1e-6)

    def _step_once(frame_idx: int) -> None:
        """Runs one frame inference and writes qpos into MuJoCo state."""
        frame = torch.from_numpy(features[frame_idx]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(frame)
        recon = outputs["x_hat"].detach().cpu().numpy().reshape(-1)
        recon_qpos = extract_qpos_from_reconstruction(
            recon=recon,
            qpos_dim=qpos_dim,
            qpos_slice_start=int(args.qpos_slice_start),
        )
        reconstructed_qpos[frame_idx] = recon_qpos

        if args.write_source == "gt":
            write_qpos = qpos_gt[frame_idx, :qpos_dim]
        else:
            write_qpos = recon_qpos
        mj_data.qpos[args.qpos_start_idx : args.qpos_start_idx + qpos_dim] = write_qpos[isaac_sim2mujoco_index]
        mujoco.mj_forward(mj_model, mj_data)

    if args.headless:
        for index in range(features.shape[0]):
            _step_once(index)
        print(f"Headless eval finished. Frames: {features.shape[0]}")
    else:
        with viewer.launch_passive(mj_model, mj_data) as gui:
            for index in range(features.shape[0]):
                start_time = time.perf_counter()
                _step_once(index)
                gui.sync()
                elapsed = time.perf_counter() - start_time
                time.sleep(max(0.0, frame_dt - elapsed))

    if args.save_recon_qpos:
        output_path = Path(args.save_recon_qpos).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(output_path),
            recon_qpos=reconstructed_qpos,
            gt_qpos=qpos_gt[:, :qpos_dim],
            motion_file=str(motion_path),
        )
        print(f"Saved reconstructed qpos to: {output_path}")


if __name__ == "__main__":
    main()
