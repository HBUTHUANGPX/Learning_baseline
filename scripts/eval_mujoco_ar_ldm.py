"""MuJoCo playback for AR-LDM rollout results.

Author: HuangPeixin
Last Modified: 2026-03-09

Purpose:
Load ``ar_ldm_rollout_eval.pt`` produced by ``scripts/eval_ar_ldm_rollout.py``,
extract generated feature sequence, map ``qpos_key`` slice with the exact
feature-concatenation logic used in ``MotionFrameDataset``, and play in MuJoCo.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modules.motion_load import MotionDatasetConfig, MotionFrameDataset


def _flatten_temporal_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Flattens a time-major tensor to shape ``[T, D]``.

    Args:
        tensor: Input tensor with temporal axis at dim-0.

    Returns:
        Flattened float tensor with shape ``[T, D]``.
    """
    if tensor.ndim == 1:
        return tensor.float().unsqueeze(-1)
    return tensor.float().reshape(tensor.shape[0], -1)


def _load_rollout_payload(path: Path) -> dict[str, object]:
    """Loads and validates rollout PT payload.

    Args:
        path: Rollout PT path.

    Returns:
        Decoded payload dictionary.
    """
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict) or "meta" not in payload or "items" not in payload:
        raise TypeError("rollout_pt must be a dict with 'meta' and 'items'.")
    items = payload["items"]
    if not isinstance(items, dict) or not items:
        raise ValueError("rollout_pt has no items.")
    return payload


def _select_item(
    payload: dict[str, object],
    npz_name: str,
    item_index: int,
) -> tuple[str, dict[str, object]]:
    """Selects one rollout item by name or index.

    Args:
        payload: Rollout payload.
        npz_name: Optional npz filename key.
        item_index: Fallback index when ``npz_name`` is empty.

    Returns:
        Tuple ``(item_key, item_value)``.
    """
    items = payload["items"]
    assert isinstance(items, dict)
    keys = list(items.keys())
    if npz_name:
        if npz_name not in items:
            raise KeyError(f"npz name '{npz_name}' not found in rollout items.")
        key = npz_name
    else:
        if item_index < 0 or item_index >= len(keys):
            raise IndexError(f"item_index out of range: {item_index}, total={len(keys)}.")
        key = keys[item_index]
    value = items[key]
    if not isinstance(value, dict):
        raise TypeError("rollout item must be a dict.")
    return key, value


def _resolve_feature_slices(
    npz_path: Path,
    feature_keys: tuple[str, ...],
    frame_stride: int,
) -> tuple[dict[str, tuple[int, int]], float]:
    """Builds feature key slices with MotionFrameDataset feature logic.

    Args:
        npz_path: Source NPZ path associated with the rollout item.
        feature_keys: Ordered concatenated feature keys.
        frame_stride: Temporal frame stride.

    Returns:
        Tuple ``(slice_map, fps)`` where ``slice_map[key]=(start, end)``.
    """
    dataset = MotionFrameDataset(
        MotionDatasetConfig(
            motion_files=(str(npz_path),),
            feature_keys=feature_keys,
            frame_stride=frame_stride,
            normalize=False,
            cache_device="cpu",
            history_frames=0,
            future_frames=0,
        )
    )

    slice_map: dict[str, tuple[int, int]] = {}
    offset = 0
    for key in feature_keys:
        member = dataset._get_member_sequence_tensor(key, sequence_id=0)
        part = _flatten_temporal_tensor(member)
        if frame_stride > 1:
            part = part[::frame_stride]
        width = int(part.shape[1])
        slice_map[key] = (offset, offset + width)
        offset += width

    with np.load(str(npz_path)) as data:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 30.0
    return slice_map, fps, dataset


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for AR-LDM MuJoCo playback."""
    parser = argparse.ArgumentParser(description="MuJoCo playback for AR-LDM rollout PT.")
    parser.add_argument("--rollout-pt", type=str, required=True)
    parser.add_argument("--mjcf", type=str, required=True)
    parser.add_argument("--rollout-npz-name", type=str, default="")
    parser.add_argument("--item-index", type=int, default=0)
    parser.add_argument("--qpos-key", type=str, default="joint_pos")
    parser.add_argument("--qpos-start-idx", type=int, default=0)
    parser.add_argument("--qpos-slice-start", type=int, default=0)
    parser.add_argument("--qpos-dim", type=int, default=0)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Runs playback for one rollout item."""
    args = parse_args()
    rollout_pt = Path(args.rollout_pt).expanduser().resolve()
    if not rollout_pt.is_file():
        raise FileNotFoundError(f"rollout_pt not found: {rollout_pt}")

    payload = _load_rollout_payload(rollout_pt)
    item_key, item = _select_item(payload, args.rollout_npz_name.strip(), int(args.item_index))
    meta = payload["meta"]
    assert isinstance(meta, dict)

    generated = item.get("generated_sequence")
    if not isinstance(generated, torch.Tensor):
        raise TypeError("generated_sequence must be a tensor.")
    generated_np = generated.detach().cpu().numpy().astype(np.float32)

    npz_path = Path(str(item.get("npz_path", ""))).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"npz_path in rollout item does not exist: {npz_path}")

    feature_keys = tuple(str(v) for v in meta.get("feature_keys", []))
    if not feature_keys:
        raise ValueError("rollout meta.feature_keys is empty.")
    frame_stride = int(meta.get("frame_stride", 1))
    slice_map, fps_from_npz, dataset = _resolve_feature_slices(npz_path, feature_keys, frame_stride)
    if args.qpos_key not in slice_map:
        raise KeyError(
            f"qpos_key '{args.qpos_key}' not in feature keys: {list(slice_map.keys())}"
        )
    q_start, q_end = slice_map[args.qpos_key]
    qpos_all = generated_np[:, q_start:q_end]

    qpos_slice_start = int(args.qpos_slice_start)
    if qpos_slice_start < 0 or qpos_slice_start >= qpos_all.shape[1]:
        raise ValueError(
            f"qpos_slice_start={qpos_slice_start} out of range for qpos dim={qpos_all.shape[1]}."
        )
    qpos_dim = int(args.qpos_dim) if args.qpos_dim > 0 else int(qpos_all.shape[1] - qpos_slice_start)
    qpos_slice_end = qpos_slice_start + qpos_dim
    if qpos_slice_end > qpos_all.shape[1]:
        raise ValueError(
            f"Requested qpos slice [{qpos_slice_start}:{qpos_slice_end}] "
            f"exceeds qpos dim {qpos_all.shape[1]}."
        )
    qpos_seq = qpos_all[:, qpos_slice_start:qpos_slice_end]

    fps = float(args.fps) if args.fps > 0 else float(fps_from_npz)
    dt = 1.0 / max(fps, 1e-6)

    import mujoco
    from mujoco import viewer

    mj_model = mujoco.MjModel.from_xml_path(str(Path(args.mjcf).expanduser().resolve()))
    mj_data = mujoco.MjData(mj_model)
    if args.qpos_start_idx + qpos_dim > mj_data.qpos.shape[0]:
        raise ValueError("qpos write range exceeds MuJoCo qpos size.")

    print(f"rollout_item={item_key}")
    print(f"npz_path={npz_path}")
    print(f"feature_dim={generated_np.shape[1]}")
    print(f"frames={qpos_seq.shape[0]}")
    print(f"qpos_key={args.qpos_key} qpos_dim={qpos_dim}")

    def _step(frame_id: int) -> None:
        mj_data.qpos[7:] = dataset.joint_pos[frame_id][dataset.isaac_sim2mujoco_index]
        # mj_data.qpos[7:] = qpos_seq[frame_id][dataset.isaac_sim2mujoco_index]
        mujoco.mj_forward(mj_model, mj_data)

    if args.headless:
        for frame_id in range(qpos_seq.shape[0]):
            _step(frame_id)
        return

    with viewer.launch_passive(mj_model, mj_data) as gui:
        for frame_id in range(qpos_seq.shape[0]):
            start_t = time.perf_counter()
            _step(frame_id)
            gui.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - start_t)))


if __name__ == "__main__":
    main()
