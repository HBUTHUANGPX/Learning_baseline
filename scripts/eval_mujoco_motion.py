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

from modules.motion_load import MotionDatasetConfig, MotionFrameDataset
from modules.vqvae import FrameFSQVAE, FrameVQVAE
from utils.math import quat_from_euler_xyz
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


def _flatten_temporal_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Flattens a temporal tensor to ``[T, D]``.

    Args:
        tensor: Input tensor with temporal axis at dim-0.

    Returns:
        Flattened tensor.
    """
    if tensor.ndim == 1:
        return tensor.float().unsqueeze(-1)
    return tensor.float().reshape(tensor.shape[0], -1)


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
) -> tuple[np.ndarray, np.ndarray, float, MotionFrameDataset]:
    """Loads model input feature array and qpos target array.

    Args:
        path: Motion NPZ path.
        feature_keys: Feature keys for model input.
        qpos_key: Key for qpos reference.
        stride: Frame sampling stride.

    Returns:
        Tuple of ``(features, qpos, fps)``.
    """
    dataset = MotionFrameDataset(
        MotionDatasetConfig(
            motion_files=(str(path),),
            feature_keys=feature_keys,
            frame_stride=stride,
            normalize=False,
            cache_device="cpu",
            history_frames=0,
            future_frames=0,
        )
    )
    features = dataset._sequence_feature_tensors[0].detach().cpu().numpy().astype(np.float32)

    if not hasattr(dataset, qpos_key):
        raise KeyError(
            f"qpos_key '{qpos_key}' is not available in dataset members. "
            "If it is derived, ensure motion_load builds it."
        )
    qpos_tensor = dataset._get_member_sequence_tensor(qpos_key, sequence_id=0)
    qpos_flat = _flatten_temporal_tensor(qpos_tensor)
    if stride > 1:
        qpos_flat = qpos_flat[::stride]
    qpos = qpos_flat.detach().cpu().numpy().astype(np.float32)

    if features.shape[0] != qpos.shape[0]:
        raise ValueError(
            f"Inconsistent feature/qpos lengths after stride: {features.shape[0]} vs {qpos.shape[0]}"
        )

    with np.load(str(path)) as data:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 30.0
    return features, qpos, fps, dataset


def _build_feature_slices(
    dataset: MotionFrameDataset,
    feature_keys: tuple[str, ...],
    stride: int,
) -> dict[str, tuple[int, int]]:
    """Builds per-key feature slices in concatenated frame feature vector.

    Args:
        dataset: Motion dataset built with the same feature key order.
        feature_keys: Ordered keys used for frame concatenation.
        stride: Frame stride applied to features.

    Returns:
        Slice map ``key -> (start, end)`` in one frame feature vector.
    """
    slice_map: dict[str, tuple[int, int]] = {}
    offset = 0
    for key in feature_keys:
        member = dataset._get_member_sequence_tensor(key, sequence_id=0)
        part = _flatten_temporal_tensor(member)
        if stride > 1:
            part = part[::stride]
        width = int(part.shape[1])
        slice_map[key] = (offset, offset + width)
        offset += width
    return slice_map


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
        cond_list.append(
            history.reshape(-1)
            if history_frames > 0
            else np.zeros((0,), dtype=np.float32)
        )
        target_list.append(target.reshape(-1))
        qpos_current.append(qpos[center])

    return (
        np.stack(enc_list, axis=0).astype(np.float32),
        (
            np.stack(cond_list, axis=0).astype(np.float32)
            if history_frames > 0
            else np.zeros((len(enc_list), 0), dtype=np.float32)
        ),
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
    parser.add_argument("--use-isaac-to-mujoco-map", action="store_true")
    parser.add_argument(
        "--inverse-from-recon",
        action="store_true",
        help=(
            "Inverse-convert reconstructed feature frame to raw root pose based on "
            "motion_load derived keys and write qpos[0:7] in MuJoCo."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Runs MuJoCo sequential playback with model inference."""
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt["args"]

    motion_path = _resolve_motion_file(
        args.motion_file, args.motion_file_yaml, args.motion_group
    )
    feature_keys = _to_tuple(args.motion_feature_keys)
    features, qpos_gt, fps, motion_dataset = _load_arrays(
        motion_path, feature_keys, args.qpos_key, args.frame_stride
    )
    feature_slices = _build_feature_slices(
        dataset=motion_dataset,
        feature_keys=feature_keys,
        stride=args.frame_stride,
    )

    history_frames = int(train_args.get("history_frames", 0))
    future_frames = int(train_args.get("future_frames", 0))
    encoder_inputs, decoder_conditions, target_windows, qpos_current = (
        _build_eval_samples(
            features=features,
            qpos=qpos_gt,
            history_frames=history_frames,
            future_frames=future_frames,
        )
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
    root_xy_origin = None
    state: dict[str, float] = {}
    if args.inverse_from_recon:
        if not hasattr(motion_dataset, "root_pos_w") or not hasattr(motion_dataset, "root_euler_w"):
            raise AttributeError(
                "inverse_from_recon requires root_pos_w and root_euler_w members from motion dataset."
            )
        root_pos = motion_dataset._get_member_sequence_tensor("root_pos_w", sequence_id=0)
        root_euler = motion_dataset._get_member_sequence_tensor("root_euler_w", sequence_id=0)
        root_xy_origin = root_pos[0, 0:2].detach().cpu().numpy().astype(np.float32)*0
        # Playback starts from center=history_frames.
        start_center = int(history_frames)
        state["yaw"] = float(root_euler[start_center, 2].item())*0

    def _step(eval_index: int) -> None:
        """Runs one inference step and writes qpos to MuJoCo."""
        enc = (
            torch.from_numpy(encoder_inputs[eval_index]).float().unsqueeze(0).to(device)
        )
        cond = (
            torch.from_numpy(decoder_conditions[eval_index])
            .float()
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            x_hat = model(enc, cond)["x_hat"].detach().cpu().numpy().reshape(-1)
        target_window = x_hat.reshape(1 + future_frames, features.shape[1])
        curr_feature = target_window[0]
        start = int(args.qpos_slice_start)
        stop = start + qpos_dim
        # Reconstruction target is [current, future], so this slice is applied
        # on reconstructed current frame by default when qpos_slice_start is set
        # within the first frame block.
        recon_qpos = x_hat[start:stop]
        write_qpos = (
            qpos_current[eval_index, :qpos_dim]
            if args.write_source == "gt"
            else recon_qpos
        )
        if args.write_source == "gt":
            print(f"Writing GT qpos for eval index {eval_index}: {write_qpos}")
        if args.use_isaac_to_mujoco_map:
            # print("Using Isaac to MuJoCo joint mapping.")
            if not hasattr(motion_dataset, "isaac_sim2mujoco_index"):
                raise AttributeError(
                    "motion_dataset missing isaac_sim2mujoco_index for joint remapping."
                )
            map_idx = motion_dataset.isaac_sim2mujoco_index
            if len(map_idx) != write_qpos.shape[0]:
                raise ValueError(
                    f"isaac->mujoco map length {len(map_idx)} does not match qpos dim {write_qpos.shape[0]}."
                )
            write_qpos = write_qpos[np.asarray(map_idx, dtype=np.int64)]

        # Optional inverse conversion from reconstructed derived features to root qpos.
        if args.inverse_from_recon:
            required = (
                "root_xy_pos",
                "root_height",
                "continuous_trigonometric_encoding",
                "delta_yaw",
            )
            if all(key in feature_slices for key in required):
                xy_s, xy_e = feature_slices["root_xy_pos"]
                h_s, h_e = feature_slices["root_height"]
                c_s, c_e = feature_slices["continuous_trigonometric_encoding"]
                d_s, d_e = feature_slices["delta_yaw"]
                root_xy_rel = curr_feature[xy_s:xy_e]
                root_h = curr_feature[h_s:h_e]
                cte = curr_feature[c_s:c_e]
                delta_yaw = curr_feature[d_s:d_e]
                if cte.shape[0] >= 4 and root_xy_origin is not None and "yaw" in state:
                    roll = float(np.arctan2(cte[0], cte[1] + 1.0))
                    pitch = float(np.arctan2(cte[2], cte[3] + 1.0))
                    yaw = float(state["yaw"])
                    root_xyz = np.array(
                        [root_xy_origin[0] + root_xy_rel[0], root_xy_origin[1] + root_xy_rel[1], root_h[0]],
                        dtype=np.float32,
                    )
                    quat = (
                        quat_from_euler_xyz(
                            torch.tensor([roll], dtype=torch.float32),
                            torch.tensor([pitch], dtype=torch.float32),
                            torch.tensor([yaw], dtype=torch.float32),
                        )[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    mj_data.qpos[0:3] = root_xyz
                    mj_data.qpos[3:7] = quat
                    state["yaw"] = yaw + float(delta_yaw[0])

        mj_data.qpos[7:] = write_qpos
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
