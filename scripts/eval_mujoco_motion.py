"""MuJoCo playback evaluation for context-conditioned motion VQ/FSQ models."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modules.vqvae import FrameFSQVAE, FrameVQVAE
from utils.load_motion_file import collect_npz_paths


def _to_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    """Converts CSV-like value to tuple.

    Args:
        value: String or iterable value.

    Returns:
        Tuple of non-empty string items.
    """
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def _flatten(array: np.ndarray) -> np.ndarray:
    """Flattens a time-major array to ``[T, D]``.

    Args:
        array: Input array.

    Returns:
        Flattened array.
    """
    if array.ndim == 1:
        return array.astype(np.float32)[:, None]
    return array.astype(np.float32).reshape(array.shape[0], -1)


def _resolve_motion_file(motion_file: str, motion_yaml: str, motion_group: str) -> Path:
    """Resolves a single NPZ motion file path.

    Args:
        motion_file: Direct file path.
        motion_yaml: YAML declaration path.
        motion_group: Group in YAML.

    Returns:
        Resolved motion file path.
    """
    if motion_file:
        path = Path(motion_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Motion file not found: {path}")
        return path

    groups = collect_npz_paths(motion_yaml)
    if motion_group:
        if motion_group not in groups:
            raise KeyError(
                f"Group '{motion_group}' not found in {motion_yaml}. "
                f"Available: {list(groups.keys())}"
            )
        files = groups[motion_group]
    else:
        files = next(iter(groups.values()))
    if not files:
        raise ValueError("No motion files resolved from yaml configuration.")
    return Path(files[0]).expanduser().resolve()


def _load_arrays(
    path: Path,
    feature_keys: tuple[str, ...],
    qpos_key: str,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Loads model input feature array and qpos target array.

    Args:
        path: Motion NPZ path.
        feature_keys: Feature keys for model input.
        qpos_key: Key for qpos reference.
        stride: Frame sampling stride.

    Returns:
        Tuple of ``(features, qpos, fps)``.
    """
    with np.load(str(path)) as data:
        parts: list[np.ndarray] = []
        lengths: list[int] = []
        for key in feature_keys:
            if key not in data:
                raise KeyError(f"Feature key '{key}' not found in {path}")
            part = _flatten(data[key])
            parts.append(part)
            lengths.append(part.shape[0])
        if qpos_key not in data:
            raise KeyError(f"qpos_key '{qpos_key}' not found in {path}")
        qpos = _flatten(data[qpos_key])
        lengths.append(qpos.shape[0])
        if len(set(lengths)) != 1:
            raise ValueError(f"Inconsistent sequence lengths in {path}: {lengths}")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 30.0

    features = np.concatenate(parts, axis=1)
    if stride > 1:
        features = features[::stride]
        qpos = qpos[::stride]
    return features, qpos, fps


def _build_eval_samples(
    features: np.ndarray,
    qpos: np.ndarray,
    history_frames: int,
    future_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds eval inputs with the same convention as training dataset.

    Args:
        features: Frame features with shape ``[T, D]``.
        qpos: Ground-truth qpos with shape ``[T, Dq]``.
        history_frames: Number of history frames.
        future_frames: Number of future frames.

    Returns:
        Tuple ``(encoder_inputs, decoder_conditions, target_windows, qpos_current)``.
    """
    min_center = int(history_frames)
    max_center = int(features.shape[0]) - 1 - int(future_frames)
    if max_center < min_center:
        raise ValueError("No valid centers for requested history/future settings.")

    centers = np.arange(min_center, max_center + 1, dtype=np.int64)
    enc_list = []
    cond_list = []
    target_list = []
    qpos_current = []

    frame_dim = int(features.shape[1])
    for center in centers:
        window = features[center - history_frames : center + future_frames + 1]
        history = features[center - history_frames : center]
        target = features[center : center + future_frames + 1]

        enc_list.append(window.reshape(-1))
        cond_list.append(history.reshape(-1) if history_frames > 0 else np.zeros((0,), dtype=np.float32))
        target_list.append(target.reshape(-1))
        qpos_current.append(qpos[center])

    return (
        np.stack(enc_list, axis=0).astype(np.float32),
        np.stack(cond_list, axis=0).astype(np.float32) if history_frames > 0 else np.zeros((len(enc_list), 0), dtype=np.float32),
        np.stack(target_list, axis=0).astype(np.float32),
        np.stack(qpos_current, axis=0).astype(np.float32),
    )


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate motion VQ/FSQ in MuJoCo.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--motion-file", default="")
    parser.add_argument("--motion-file-yaml", default="configs/data/motion_file.yaml")
    parser.add_argument("--motion-group", default="")
    parser.add_argument("--motion-feature-keys", default="joint_pos,joint_vel")
    parser.add_argument("--qpos-key", default="joint_pos")
    parser.add_argument("--qpos-start-idx", type=int, default=0)
    parser.add_argument("--qpos-slice-start", type=int, default=0)
    parser.add_argument("--qpos-dim", type=int, default=0)
    parser.add_argument("--write-source", choices=["recon", "gt"], default="recon")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Runs MuJoCo sequential playback with model inference."""
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt["args"]

    motion_path = _resolve_motion_file(args.motion_file, args.motion_file_yaml, args.motion_group)
    feature_keys = _to_tuple(args.motion_feature_keys)
    features, qpos_gt, fps = _load_arrays(motion_path, feature_keys, args.qpos_key, args.frame_stride)

    history_frames = int(train_args.get("history_frames", 0))
    future_frames = int(train_args.get("future_frames", 0))
    encoder_inputs, decoder_conditions, target_windows, qpos_current = _build_eval_samples(
        features=features,
        qpos=qpos_gt,
        history_frames=history_frames,
        future_frames=future_frames,
    )

    encoder_input_dim = int(ckpt.get("encoder_input_dim", encoder_inputs.shape[1]))
    decoder_condition_dim = int(
        ckpt.get("decoder_condition_dim", decoder_conditions.shape[1])
    )
    target_dim = int(ckpt.get("target_dim", target_windows.shape[1]))

    if train_args["model"] == "vq":
        model = FrameVQVAE(
            encoder_input_dim=encoder_input_dim,
            decoder_condition_dim=decoder_condition_dim,
            target_dim=target_dim,
            embedding_dim=int(train_args["embedding_dim"]),
            hidden_dim=int(train_args["hidden_dim"]),
            num_embeddings=int(train_args["num_embeddings"]),
            beta=float(train_args["beta"]),
            recon_loss_mode=str(train_args["recon_loss_mode"]),
        )
    else:
        model = FrameFSQVAE(
            encoder_input_dim=encoder_input_dim,
            decoder_condition_dim=decoder_condition_dim,
            target_dim=target_dim,
            embedding_dim=int(train_args["embedding_dim"]),
            hidden_dim=int(train_args["hidden_dim"]),
            fsq_levels=int(train_args["fsq_levels"]),
            beta=float(train_args["beta"]),
            recon_loss_mode=str(train_args["recon_loss_mode"]),
        )

    model.load_state_dict(ckpt["model_state"], strict=True)
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    qpos_dim = int(args.qpos_dim) if args.qpos_dim > 0 else int(qpos_current.shape[1])

    import mujoco
    from mujoco import viewer

    mj_model = mujoco.MjModel.from_xml_path(str(Path(args.mjcf).expanduser().resolve()))
    mj_data = mujoco.MjData(mj_model)

    if args.qpos_start_idx + qpos_dim > mj_data.qpos.shape[0]:
        raise ValueError("qpos write range exceeds MuJoCo qpos size.")

    dt = 1.0 / max(float(fps), 1e-6)

    def _step(eval_index: int) -> None:
        """Runs one inference step and writes qpos to MuJoCo."""
        enc = torch.from_numpy(encoder_inputs[eval_index]).float().unsqueeze(0).to(device)
        cond = torch.from_numpy(decoder_conditions[eval_index]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            x_hat = model(enc, cond)["x_hat"].detach().cpu().numpy().reshape(-1)
        start = int(args.qpos_slice_start)
        stop = start + qpos_dim
        # Reconstruction target is [current, future], so this slice is applied
        # on reconstructed current frame by default when qpos_slice_start is set
        # within the first frame block.
        recon_qpos = x_hat[start:stop]
        write_qpos = qpos_current[eval_index, :qpos_dim] if args.write_source == "gt" else recon_qpos
        mj_data.qpos[args.qpos_start_idx : args.qpos_start_idx + qpos_dim] = write_qpos
        mujoco.mj_forward(mj_model, mj_data)

    if args.headless:
        for index in range(encoder_inputs.shape[0]):
            _step(index)
        return

    with viewer.launch_passive(mj_model, mj_data) as gui:
        for index in range(encoder_inputs.shape[0]):
            start_t = time.perf_counter()
            _step(index)
            gui.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - start_t)))


if __name__ == "__main__":
    main()
