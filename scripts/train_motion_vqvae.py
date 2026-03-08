"""Training entrypoint for context-conditioned motion VQ-VAE and FSQ-VAE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch
from torch.optim import Adam
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modules.data import DataConfig, create_motion_dataloaders
from modules.vqvae import FrameFSQVAE, FrameVQVAE
from utils import TensorboardLogger, create_experiment_paths, set_seed


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for motion VQ/FSQ training.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train context-conditioned motion VQ/FSQ."
    )
    parser.add_argument("--model", choices=["vq", "fsq"], default="fsq")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-embeddings", type=int, default=512)
    parser.add_argument("--fsq-levels", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument(
        "--recon-loss-mode", choices=["bce", "mse"], default="mse"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--motion-files", type=str, default="")
    parser.add_argument(
        "--motion-file-yaml", type=str, default="configs/data/motion_file.yaml"
    )
    parser.add_argument("--motion-group", type=str, default="")
    parser.add_argument("--motion-feature-keys", type=str, default="joint_pos,joint_vel")
    parser.add_argument("--motion-frame-stride", type=int, default=1)
    parser.add_argument("--motion-normalize", action="store_true")
    parser.add_argument(
        "--motion-cache-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Where to cache dataset tensors. Use 'cuda' to avoid per-batch H2D copies.",
    )
    parser.add_argument("--history-frames", type=int, default=0)
    parser.add_argument("--future-frames", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--log-root", type=str, default="./log")
    return parser.parse_args()


def _to_tuple(value: str) -> tuple[str, ...]:
    """Converts comma-separated text into string tuple.

    Args:
        value: Comma-separated string.

    Returns:
        Clean tuple of non-empty items.
    """
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _select_device(raw: str) -> torch.device:
    """Resolves runtime device from user selection.

    Args:
        raw: Device option from CLI.

    Returns:
        Torch device object.
    """
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(
    args: argparse.Namespace,
    encoder_input_dim: int,
    decoder_condition_dim: int,
    target_dim: int,
) -> torch.nn.Module:
    """Builds VQ or FSQ model from training arguments.

    Args:
        args: Training argument namespace.
        encoder_input_dim: Encoder input dimension inferred from dataset.
        decoder_condition_dim: Decoder condition dimension inferred from dataset.
        target_dim: Reconstruction target dimension inferred from dataset.

    Returns:
        Constructed model instance.
    """
    if args.model == "vq":
        return FrameVQVAE(
            encoder_input_dim=encoder_input_dim,
            decoder_condition_dim=decoder_condition_dim,
            target_dim=target_dim,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_embeddings=args.num_embeddings,
            beta=args.beta,
            recon_loss_mode=args.recon_loss_mode,
        )
    return FrameFSQVAE(
        encoder_input_dim=encoder_input_dim,
        decoder_condition_dim=decoder_condition_dim,
        target_dim=target_dim,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        fsq_levels=args.fsq_levels,
        recon_loss_mode=args.recon_loss_mode,
    )


def _run_epoch(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> Dict[str, float]:
    """Runs one epoch for train or validation mode.

    Args:
        model: Model object.
        loader: Dataloader instance.
        device: Runtime device.
        optimizer: Optimizer for training. Use ``None`` for validation.

    Returns:
        Aggregated metric dictionary.
    """
    is_train = optimizer is not None
    model.train(is_train)

    metric_sum: dict[str, float] = {}
    sample_count = 0

    progress = tqdm(loader, desc="Train" if is_train else "Val", leave=False)
    for batch in progress:
        encoder_input = batch["encoder_input"]
        decoder_condition = batch["decoder_condition"]
        target = batch["target"]
        if encoder_input.device != device:
            encoder_input = encoder_input.to(device, non_blocking=True)
        if decoder_condition.device != device:
            decoder_condition = decoder_condition.to(device, non_blocking=True)
        if target.device != device:
            target = target.to(device, non_blocking=True)

        outputs = model(encoder_input, decoder_condition)
        losses = model.loss_function(target, outputs)

        if is_train:
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

        batch_size = int(encoder_input.shape[0])
        sample_count += batch_size
        for key, value in losses.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.numel() != 1:
                continue
            metric_sum[key] = metric_sum.get(key, 0.0) + float(value.detach().cpu()) * batch_size

    if sample_count == 0:
        return {"loss": 0.0, "recon_loss": 0.0, "quant_loss": 0.0}
    if "loss" not in metric_sum:
        metric_sum["loss"] = 0.0
    if "recon_loss" not in metric_sum:
        metric_sum["recon_loss"] = 0.0
    if "quant_loss" not in metric_sum:
        metric_sum["quant_loss"] = 0.0
    return {key: value / sample_count for key, value in metric_sum.items()}


def main(args: argparse.Namespace | None = None) -> None:
    """Executes end-to-end motion VQ/FSQ training.

    Args:
        args: Optional prebuilt argument namespace.
    """
    if args is None:
        args = parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    device = _select_device(args.device)
    paths = create_experiment_paths(log_root=args.log_root)
    tb_logger = TensorboardLogger(paths.tensorboard_dir)

    motion_cache_device = getattr(args, "motion_cache_device", "auto")
    data_config = DataConfig(
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        motion_files=_to_tuple(args.motion_files),
        motion_file_yaml=args.motion_file_yaml,
        motion_group=args.motion_group,
        motion_feature_keys=_to_tuple(args.motion_feature_keys),
        motion_frame_stride=args.motion_frame_stride,
        motion_normalize=args.motion_normalize,
        motion_cache_device=str(device) if motion_cache_device == "auto" else motion_cache_device,
        history_frames=args.history_frames,
        future_frames=args.future_frames,
    )
    (
        train_loader,
        val_loader,
        encoder_input_dim,
        decoder_condition_dim,
        target_dim,
    ) = create_motion_dataloaders(data_config)

    model = _build_model(
        args,
        encoder_input_dim=encoder_input_dim,
        decoder_condition_dim=decoder_condition_dim,
        target_dim=target_dim,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, device, None)

        tb_logger.log_scalars(train_metrics, epoch, prefix="train")
        tb_logger.log_scalars(val_metrics, epoch, prefix="val")

        checkpoint_path = paths.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "args": vars(args),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "encoder_input_dim": encoder_input_dim,
                "decoder_condition_dim": decoder_condition_dim,
                "target_dim": target_dim,
            },
            checkpoint_path,
        )
        print(
            f"[Epoch {epoch}] train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f}"
        )

    tb_logger.close()
    print(f"Training finished. Logs saved in: {paths.run_dir}")


if __name__ == "__main__":
    main()
