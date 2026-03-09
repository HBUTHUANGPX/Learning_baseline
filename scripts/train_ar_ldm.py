"""Train AR-LDM on FSQ latent targets with text and motion conditions.
train_ar_ldm.py
Author: HuangPeixin
Last Modified: 2026-03-09

Pipeline relation:
1. Read latent-condition batches from ``modules.latent_data``.
2. Train ``modules.ar_ldm.ARLDMTransformer`` with DDPM objective in latent space.
3. Save checkpoints for later autoregressive rollout with frozen FSQ decoder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import torch
from tqdm.auto import tqdm

try:
    from omegaconf import OmegaConf
except Exception as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "OmegaConf is required. Install via `pip install omegaconf`."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modules.ar_ldm import ARLDMConfig, ARLDMTransformer
from modules.latent_data import LatentDataConfig, create_latent_condition_loaders
from utils import TensorboardLogger, create_experiment_paths, set_seed


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for AR-LDM training.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(description="Train AR-LDM on FSQ latent targets.")
    parser.add_argument("--config", type=str, default="configs/ar_ldm.yaml")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. model.lr=1e-4 train.epochs=100",
    )
    return parser.parse_args()


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Moves tensor batch fields to target device.

    Args:
        batch: Input batch dictionary.
        device: Target runtime device.

    Returns:
        Device-mapped batch dictionary.
    """
    moved: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True)
    return moved


def _shard_indices_for_ddp(indices: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Shards index tensor for DDP-style process-local iteration.

    Args:
        indices: Global sample index tensor.
        rank: Current process rank.
        world_size: Number of distributed processes.

    Returns:
        Rank-specific index tensor view.
    """
    if world_size <= 1:
        return indices
    return indices[rank::world_size].contiguous()


def _run_epoch(
    model: ARLDMTransformer,
    loader: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, float]:
    """Runs one train/validation epoch.

    Args:
        model: AR-LDM model.
        loader: Batch loader iterable.
        optimizer: Optimizer in training mode, ``None`` for validation.
        device: Runtime device.

    Returns:
        Averaged scalar metrics for the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)
    metric_sum: Dict[str, float] = {}
    sample_count = 0

    progress = tqdm(loader, desc="Train" if is_train else "Val", leave=False)
    for batch in progress:
        batch_d = _to_device(batch, device=device)
        losses = model.training_step(batch_d)

        if is_train:
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

        bsz = int(batch_d["target_latent"].shape[0])
        sample_count += bsz
        for key, value in losses.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metric_sum[key] = metric_sum.get(key, 0.0) + float(value.detach().cpu()) * bsz

    if sample_count == 0:
        return {"loss": 0.0, "diffusion_loss": 0.0}
    return {key: value / sample_count for key, value in metric_sum.items()}


def main(args: argparse.Namespace | None = None) -> None:
    """Executes AR-LDM training loop from OmegaConf config.

    Args:
        args: Optional prebuilt argument namespace.
    """
    if args is None:
        args = parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    set_seed(int(cfg.train.seed), deterministic=bool(cfg.train.deterministic))
    device = torch.device(cfg.train.device)

    latent_data_cfg = LatentDataConfig(
        batch_size=int(cfg.data.batch_size),
        val_ratio=float(cfg.data.val_ratio),
        seed=int(cfg.train.seed),
        motion_files=tuple(cfg.data.motion_files),
        motion_file_yaml=str(cfg.data.motion_file_yaml),
        motion_group=str(cfg.data.motion_group),
        motion_feature_keys=tuple(cfg.data.motion_feature_keys),
        motion_frame_stride=int(cfg.data.motion_frame_stride),
        motion_normalize=bool(cfg.data.motion_normalize),
        motion_cache_device=str(cfg.data.motion_cache_device),
        history_frames=int(cfg.data.history_frames),
        future_frames=int(cfg.data.future_frames),
        text_token_pt=str(cfg.data.text_token_pt),
        fsq_checkpoint=str(cfg.data.fsq_checkpoint),
        fsq_device=str(cfg.data.fsq_device),
    )
    train_loader, val_loader, meta = create_latent_condition_loaders(latent_data_cfg)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = int(torch.distributed.get_rank())
        world_size = int(torch.distributed.get_world_size())
        train_loader.dataset.indices = _shard_indices_for_ddp(train_loader.dataset.indices, rank, world_size)
        val_loader.dataset.indices = _shard_indices_for_ddp(val_loader.dataset.indices, rank, world_size)

    model_cfg = ARLDMConfig(
        latent_dim=int(meta["latent_dim"]),
        cond_motion_dim=int(meta["cond_motion_dim"]),
        text_dim=int(meta["text_dim"]),
        model_dim=int(cfg.model.model_dim),
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        dropout=float(cfg.model.dropout),
        timesteps=int(cfg.model.timesteps),
        beta_start=float(cfg.model.beta_start),
        beta_end=float(cfg.model.beta_end),
        backbone_type=str(cfg.model.backbone_type),
        fusion_type=str(cfg.model.fusion_type),
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
    )
    model = ARLDMTransformer(model_cfg).to(device)
    print("Model architecture:", model)
    optimizer, scheduler = model.configure_optimizers()

    paths = create_experiment_paths(log_root=str(cfg.log.root))
    tb_logger = TensorboardLogger(paths.tensorboard_dir)

    for epoch in range(1, int(cfg.train.epochs) + 1):
        train_metrics = _run_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device)
        val_metrics = _run_epoch(model=model, loader=val_loader, optimizer=None, device=device)
        scheduler.step()

        tb_logger.log_scalars(train_metrics, epoch, prefix="train")
        tb_logger.log_scalars(val_metrics, epoch, prefix="val")

        ckpt_path = paths.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "meta": meta,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            ckpt_path,
        )
        print(
            f"[Epoch {epoch}] train_loss={train_metrics.get('loss', 0.0):.6f} "
            f"val_loss={val_metrics.get('loss', 0.0):.6f}"
        )

    tb_logger.close()
    print(f"Training finished. Logs saved in: {paths.run_dir}")


if __name__ == "__main__":
    main()

