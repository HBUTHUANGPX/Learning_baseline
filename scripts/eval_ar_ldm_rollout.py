"""Evaluates AR-LDM rollout with a frozen FSQ decoder on motion trajectories.

Author: HuangPeixin
Last Modified: 2026-03-09

Pipeline relation:
1. Loads AR-LDM checkpoint trained on FSQ latent targets.
2. Loads frozen FSQ decoder checkpoint.
3. Uses real trajectory history (n frames) + text token to autoregressively
   generate ``current + future`` blocks.
4. Saves rollout tensors per NPZ into a single PT artifact for downstream
   inspection/evaluation.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Mapping

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modules.ar_ldm import ARLDMConfig, ARLDMTransformer
from modules.ar_rollout import ARLatentRolloutGenerator, FrozenFSQDecoder, RolloutConfig
from modules.motion_load import MotionDatasetConfig, MotionFrameDataset
from utils import set_seed
from utils.load_motion_file import collect_npz_paths


def _parse_csv_tuple(raw: str) -> tuple[str, ...]:
    """Parses comma-separated text into a tuple of non-empty entries.

    Args:
        raw: Raw comma-separated text.

    Returns:
        Tuple of stripped non-empty strings.
    """
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _resolve_device(raw: str) -> torch.device:
    """Resolves runtime device from CLI option.

    Args:
        raw: Device option in ``{"auto", "cpu", "cuda"}``.

    Returns:
        Resolved torch device.

    Raises:
        RuntimeError: If CUDA is requested but unavailable.
        ValueError: If option is not supported.
    """
    norm = raw.lower().strip()
    if norm == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if norm == "cpu":
        return torch.device("cpu")
    if norm == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    raise ValueError(f"Unsupported device: {raw}")


def _resolve_motion_files(
    motion_file: str,
    motion_file_yaml: str,
    motion_group: str,
) -> list[Path]:
    """Resolves motion files from explicit path or grouped YAML.

    Args:
        motion_file: Optional explicit NPZ path.
        motion_file_yaml: YAML file with grouped NPZ lists.
        motion_group: Group key in YAML.

    Returns:
        List of resolved NPZ paths.

    Raises:
        ValueError: If no files can be resolved.
        FileNotFoundError: If resolved path does not exist.
        KeyError: If requested group is absent in YAML.
    """
    if motion_file.strip():
        path = Path(motion_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"motion_file not found: {path}")
        return [path]

    grouped = collect_npz_paths(motion_file_yaml)
    if motion_group:
        if motion_group not in grouped:
            raise KeyError(
                f"motion_group '{motion_group}' not found in {motion_file_yaml}. "
                f"Available groups: {list(grouped.keys())}"
            )
        files = grouped[motion_group]
    else:
        files = []
        for group_files in grouped.values():
            files.extend(group_files)
    if not files:
        raise ValueError("No motion files resolved for evaluation.")
    return [Path(file).expanduser().resolve() for file in files]


def _load_text_token_entry(
    text_token_map: Mapping[str, Mapping[str, object]],
    npz_name: str,
) -> tuple[torch.Tensor, str]:
    """Loads one trajectory-level text token and prompt string by NPZ file name.

    Args:
        text_token_map: Mapping loaded from PT file.
        npz_name: NPZ filename key.

    Returns:
        Tuple ``(token, prompt)`` where token is 1D float tensor.

    Raises:
        KeyError: If key or token field is missing.
        TypeError: If token field is not a tensor.
    """
    if npz_name not in text_token_map:
        raise KeyError(f"Text token map missing key: {npz_name}")
    entry = text_token_map[npz_name]
    if "pooler_output" in entry:
        token = entry["pooler_output"]
    elif "token" in entry:
        token = entry["token"]
    else:
        raise KeyError(
            f"Text token entry for {npz_name} must contain 'pooler_output' or 'token'."
        )
    if not isinstance(token, torch.Tensor):
        raise TypeError(f"Text token for {npz_name} must be torch.Tensor, got {type(token)}.")
    prompt = str(entry.get("clip_text_prompt", ""))
    return token.float().reshape(-1), prompt


def _build_ar_ldm_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ARLDMTransformer, Mapping[str, object], Mapping[str, int]]:
    """Builds AR-LDM model from training checkpoint.

    Args:
        checkpoint_path: AR-LDM checkpoint path.
        device: Runtime device.

    Returns:
        Tuple of ``(model, config_dict, meta_dict)``.

    Raises:
        KeyError: If checkpoint misses required fields.
    """
    state = torch.load(str(checkpoint_path), map_location="cpu")
    for key in ("config", "meta", "model_state"):
        if key not in state:
            raise KeyError(f"AR-LDM checkpoint missing key: {key}")
    cfg = state["config"]
    meta = state["meta"]

    model_cfg = ARLDMConfig(
        latent_dim=int(meta["latent_dim"]),
        cond_motion_dim=int(meta["cond_motion_dim"]),
        text_dim=int(meta["text_dim"]),
        model_dim=int(cfg["model"]["model_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        num_heads=int(cfg["model"]["num_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        timesteps=int(cfg["model"]["timesteps"]),
        beta_start=float(cfg["model"]["beta_start"]),
        beta_end=float(cfg["model"]["beta_end"]),
        backbone_type=str(cfg["model"]["backbone_type"]),
        fusion_type=str(cfg["model"]["fusion_type"]),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )
    model = ARLDMTransformer(model_cfg)
    model.load_state_dict(state["model_state"], strict=True)
    model = model.to(device).eval()
    return model, cfg, meta


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate AR-LDM rollout.")
    parser.add_argument("--ar-ldm-ckpt", type=str, required=True)
    parser.add_argument("--fsq-ckpt", type=str, required=True)
    parser.add_argument("--text-token-pt", type=str, required=True)
    parser.add_argument("--motion-file", type=str, default="")
    parser.add_argument("--motion-file-yaml", type=str, default="configs/data/motion_file.yaml")
    parser.add_argument("--motion-group", type=str, default="")
    parser.add_argument("--motion-feature-keys", type=str, default="")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--seed-mode", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", type=str, default="tmp/ar_ldm_rollout_eval.pt")
    return parser.parse_args()


def main() -> None:
    """Runs AR-LDM rollout evaluation and saves per-trajectory outputs."""
    args = parse_args()
    set_seed(int(args.seed), deterministic=False)
    random.seed(int(args.seed))
    device = _resolve_device(args.device)

    ar_ckpt_path = Path(args.ar_ldm_ckpt).expanduser().resolve()
    fsq_ckpt_path = Path(args.fsq_ckpt).expanduser().resolve()
    text_pt_path = Path(args.text_token_pt).expanduser().resolve()
    if not ar_ckpt_path.is_file():
        raise FileNotFoundError(f"AR-LDM checkpoint not found: {ar_ckpt_path}")
    if not fsq_ckpt_path.is_file():
        raise FileNotFoundError(f"FSQ checkpoint not found: {fsq_ckpt_path}")
    if not text_pt_path.is_file():
        raise FileNotFoundError(f"text-token PT not found: {text_pt_path}")

    ldm_model, train_cfg, meta = _build_ar_ldm_from_checkpoint(ar_ckpt_path, device=device)
    fsq_decoder = FrozenFSQDecoder(checkpoint_path=str(fsq_ckpt_path), device=str(device))

    history_frames = int(meta["history_frames"])
    future_frames = int(meta["future_frames"])
    if history_frames > (1 + future_frames):
        raise ValueError(
            "Invalid temporal setup from checkpoint meta: history_frames must satisfy n <= 1 + m. "
            f"Got n={history_frames}, m={future_frames}."
        )
    frame_dim = int(meta["cond_motion_dim"]) // max(history_frames, 1)
    if history_frames <= 0:
        raise ValueError("history_frames must be > 0 for autoregressive rollout.")
    if int(meta["cond_motion_dim"]) != history_frames * frame_dim:
        raise ValueError("Invalid cond_motion_dim in checkpoint meta.")

    rollout = ARLatentRolloutGenerator(
        ldm_model=ldm_model,
        fsq_decoder=fsq_decoder,
        config=RolloutConfig(
            history_frames=history_frames,
            future_frames=future_frames,
            frame_dim=frame_dim,
            diffusion_steps=int(args.diffusion_steps),
        ),
    )

    feature_keys_raw = args.motion_feature_keys.strip()
    if feature_keys_raw:
        feature_keys = _parse_csv_tuple(feature_keys_raw)
    else:
        feature_keys = tuple(train_cfg["data"]["motion_feature_keys"])
    motion_files = _resolve_motion_files(
        motion_file=args.motion_file,
        motion_file_yaml=args.motion_file_yaml,
        motion_group=args.motion_group,
    )
    if args.limit > 0:
        motion_files = motion_files[: int(args.limit)]

    text_token_map = torch.load(str(text_pt_path), map_location="cpu")
    if not isinstance(text_token_map, dict):
        raise TypeError("text-token PT must store a dict mapping npz_name -> token entry.")

    result: dict[str, object] = {
        "meta": {
            "ar_ldm_ckpt": str(ar_ckpt_path),
            "fsq_ckpt": str(fsq_ckpt_path),
            "text_token_pt": str(text_pt_path),
            "history_frames": history_frames,
            "future_frames": future_frames,
            "frame_dim": frame_dim,
            "feature_keys": list(feature_keys),
            "frame_stride": int(args.frame_stride),
            "num_steps": int(args.num_steps),
            "seed_mode": str(args.seed_mode),
            "diffusion_steps": int(args.diffusion_steps),
            "device": str(device),
        },
        "items": {},
    }

    # Reuse MotionFrameDataset feature pipeline so derived keys (e.g.
    # continuous_trigonometric_encoding) are built exactly as in training.
    motion_dataset = MotionFrameDataset(
        MotionDatasetConfig(
            motion_files=tuple(str(path) for path in motion_files),
            feature_keys=feature_keys,
            frame_stride=int(args.frame_stride),
            normalize=False,
            cache_device="cpu",
            history_frames=history_frames,
            future_frames=future_frames,
        )
    )

    progress = tqdm(range(motion_dataset.num_sequences), desc="AR-LDM rollout", unit="seq")
    generator = torch.Generator().manual_seed(int(args.seed))
    for sequence_id in progress:
        motion_path = motion_dataset.paths[sequence_id]
        sequence = motion_dataset._sequence_feature_tensors[sequence_id].detach().cpu()
        if sequence.shape[1] != frame_dim:
            raise ValueError(
                f"Feature dim mismatch for {motion_path.name}: got {sequence.shape[1]}, expected {frame_dim}."
            )
        if sequence.shape[0] < history_frames:
            raise ValueError(
                f"Sequence {motion_path.name} has only {sequence.shape[0]} frames, "
                f"but history_frames={history_frames}."
            )

        if args.seed_mode == "first":
            start = 0
        else:
            max_start = int(sequence.shape[0] - history_frames)
            start = int(torch.randint(0, max_start + 1, (1,), generator=generator).item())
        seed_history = sequence[start : start + history_frames]

        text_token, prompt = _load_text_token_entry(text_token_map, npz_name=motion_path.name)
        generated_sequence, generated_blocks = rollout.rollout(
            seed_history=seed_history,
            text_token=text_token,
            num_steps=int(args.num_steps),
        )

        result["items"][motion_path.name] = {
            "npz_path": str(motion_path),
            "start_index": int(start),
            "seed_history": seed_history.cpu(),
            "generated_sequence": generated_sequence.cpu(),
            "generated_blocks": generated_blocks.cpu(),
            "clip_text_prompt": prompt,
            "text_token": text_token.cpu(),
        }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    print(f"saved={output_path}")
    print(f"num_sequences={len(result['items'])}")


if __name__ == "__main__":
    main()
