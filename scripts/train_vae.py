"""Training script for VAE-family models with manager-style orchestration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable

import torch
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modules.algorithms import (
    AlgorithmTerm,
    VAEAlgorithmTerm,
    build_algorithm_term,
)
from modules.data import DataConfig, create_dataloader
from modules.motion_load import (
    load_motion_feature_sequence,
    resolve_motion_files,
)
from utils.load_motion_file import collect_npz_paths
from utils import (
    TensorboardLogger,
    create_experiment_paths,
    save_reconstruction_batch,
    set_seed,
)

# Backward-compatible alias for existing tests/scripts.
VAETerm = VAEAlgorithmTerm


def _init_agg(metric_keys: Iterable[str]) -> Dict[str, float]:
    """Creates zero-valued aggregator for metric keys.

    Args:
        metric_keys: Metric names to aggregate.

    Returns:
        Mapping from metric name to zero value.
    """
    return {key: 0.0 for key in metric_keys}


def _to_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    """Normalizes comma-separated string or iterable into string tuple.

    Args:
        value: Input value from CLI args/namespace.

    Returns:
        Cleaned tuple of non-empty strings.
    """
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def train_one_epoch(
    term: AlgorithmTerm,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Runs one training epoch with a model term adapter.

    Args:
        term: AlgorithmTerm adapter.
        loader: Training dataloader.
        optimizer: Torch optimizer.
        device: Computation device.

    Returns:
        Mean scalar metrics over the epoch.
    """
    term.model.train()
    agg = _init_agg(term.metric_keys)
    num_samples = 0

    for batch in loader:
        x, _, losses = term.compute(batch, device=device)
        batch_size = int(x.shape[0])
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

        num_samples += batch_size
        for key in term.metric_keys:
            # Losses are per-sample averages; convert back to sample-weighted sum.
            agg[key] += float(losses[key].detach().cpu()) * batch_size

    return {key: value / max(num_samples, 1) for key, value in agg.items()}


@torch.no_grad()
def validate_one_epoch(
    term: AlgorithmTerm,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Runs one validation epoch with a model term adapter.

    Args:
        term: AlgorithmTerm adapter.
        loader: Validation dataloader.
        device: Computation device.

    Returns:
        Mean scalar metrics over the epoch.
    """
    term.model.eval()
    agg = _init_agg(term.metric_keys)
    num_samples = 0

    for batch in loader:
        x, _, losses = term.compute(batch, device=device)
        batch_size = int(x.shape[0])
        num_samples += batch_size
        for key in term.metric_keys:
            agg[key] += float(losses[key].detach().cpu()) * batch_size

    return {key: value / max(num_samples, 1) for key, value in agg.items()}


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for training.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Train VAE-family models.")
    parser.add_argument(
        "--model",
        type=str,
        default="vanilla",
        choices=["vanilla", "beta", "conv", "vq", "fsq"],
    )
    parser.add_argument("--algorithm", type=str, default="vae", choices=["vae"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="random_binary",
        choices=["random_binary", "mnist", "random_sequence", "motion_mimic"],
    )
    parser.add_argument("--input-dim", type=int, default=784)
    parser.add_argument("--image-channels", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=28)
    parser.add_argument("--image-width", type=int, default=28)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dims", type=str, default="256,128")
    parser.add_argument("--conv-channels", type=str, default="32,64")
    parser.add_argument("--conv-bottleneck-dim", type=int, default=256)
    parser.add_argument("--vq-decoder-channels", type=str, default="64,32")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument(
        "--recon-loss-mode",
        type=str,
        default="auto",
        choices=["auto", "bce", "mse"],
    )
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--num-embeddings", type=int, default=128)
    parser.add_argument("--fsq-levels", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--sequence-feature-dim", type=int, default=16)
    parser.add_argument("--sequence-variable-length", action="store_true")
    parser.add_argument("--sequence-min-length", type=int, default=8)
    parser.add_argument("--motion-files", type=str, default="")
    parser.add_argument("--motion-file-yaml", type=str, default="")
    parser.add_argument("--motion-group", type=str, default="")
    parser.add_argument(
        "--motion-feature-keys", type=str, default="joint_pos,joint_vel"
    )
    parser.add_argument("--motion-as-sequence", action="store_true")
    parser.add_argument("--motion-frame-stride", type=int, default=1)
    parser.add_argument("--motion-normalize", action="store_true")
    parser.add_argument("--no-batch-protocol", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--log-root", type=str, default="./log")
    return parser.parse_args()


class ExperimentManager:
    """Manager-style trainer coordinating data, model term, logging, and checkpoints."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initializes manager state from CLI arguments.

        Args:
            args: Parsed command-line arguments.
        """
        self.args = args
        if args.device == "cpu":
            self.device = torch.device("cpu")
        elif args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but not available.")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paths = create_experiment_paths(log_root=args.log_root)
        self.tb_logger = TensorboardLogger(self.paths.tensorboard_dir)
        self._adapt_motion_model_input_dim(args)
        self.term = build_algorithm_term(args)
        effective_input_dim = (
            args.image_channels * args.image_height * args.image_width
            if self.term.expects_image_input
            else args.input_dim
        )

        data_config = DataConfig(
            dataset=args.dataset,
            input_dim=effective_input_dim,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            data_root=args.data_root,
            seed=args.seed,
            flatten=not self.term.expects_image_input,
            image_size=args.image_height,
            image_channels=args.image_channels,
            sequence_length=args.sequence_length,
            sequence_feature_dim=args.sequence_feature_dim,
            sequence_variable_length=args.sequence_variable_length,
            sequence_min_length=args.sequence_min_length,
            motion_files=_to_tuple(args.motion_files),
            motion_file_yaml=args.motion_file_yaml,
            motion_group=args.motion_group,
            motion_feature_keys=_to_tuple(args.motion_feature_keys),
            motion_as_sequence=args.motion_as_sequence,
            motion_frame_stride=args.motion_frame_stride,
            motion_normalize=args.motion_normalize,
            use_batch_protocol=not args.no_batch_protocol,
        )
        self.train_loader, self.val_loader = create_dataloader(data_config)
        self.term.model = self.term.model.to(self.device)
        self.optimizer = Adam(self.term.model.parameters(), lr=args.lr)

    @staticmethod
    def _adapt_motion_model_input_dim(args: argparse.Namespace) -> None:
        """Adapts model input dimensions for motion_mimic dataset.

        For frame-wise training with MLP VAE models, input dimension is inferred
        from actual motion features to avoid manual mismatch configuration.

        Args:
            args: Parsed argument namespace (mutated in-place).
        """
        if args.dataset != "motion_mimic":
            return
        if args.model in {"vanilla", "beta"} and args.motion_as_sequence:
            raise ValueError(
                "motion_mimic with motion_as_sequence=true returns [B, T, D] batches, "
                "which are incompatible with current MLP VAE models. "
                "Use data.motion_as_sequence=false, or switch to a sequence-capable algorithm."
            )
        if args.model not in {"vanilla", "beta"}:
            return

        motion_files = _to_tuple(args.motion_files)
        if not motion_files:
            if not args.motion_file_yaml:
                raise ValueError(
                    "motion_mimic requires motion_files or motion_file_yaml to infer input_dim."
                )
            grouped = collect_npz_paths(args.motion_file_yaml)
            if args.motion_group:
                if args.motion_group not in grouped:
                    raise KeyError(
                        f"Motion group '{args.motion_group}' not found in {args.motion_file_yaml}. "
                        f"Available: {list(grouped.keys())}"
                    )
                motion_files = tuple(grouped[args.motion_group])
            else:
                merged = []
                for files in grouped.values():
                    merged.extend(files)
                motion_files = tuple(merged)
        paths = resolve_motion_files(motion_files=motion_files, motion_file_group=None)
        first_feature = load_motion_feature_sequence(
            path=paths[0],
            feature_keys=_to_tuple(args.motion_feature_keys),
            frame_stride=args.motion_frame_stride,
        )
        args.input_dim = int(first_feature.shape[-1])

    def run(self) -> None:
        """Executes training loop for the configured number of epochs."""
        for epoch in range(1, self.args.epochs + 1):
            train_metrics = train_one_epoch(
                self.term, self.train_loader, self.optimizer, self.device
            )
            val_metrics = validate_one_epoch(self.term, self.val_loader, self.device)

            self.tb_logger.log_scalars(train_metrics, epoch, prefix="train")
            self.tb_logger.log_scalars(val_metrics, epoch, prefix="val")
            self._save_epoch_artifacts(epoch)

            print(
                f"[Epoch {epoch}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f}"
            )

        self.tb_logger.close()
        print(f"Training finished. Logs saved in: {self.paths.run_dir}")

    @torch.no_grad()
    def _save_epoch_artifacts(self, epoch: int) -> None:
        """Saves checkpoint and reconstruction artifact for one epoch.

        Args:
            epoch: Current epoch index (1-based).
        """
        sample_loader = self._select_sample_loader()
        raw_batch = next(iter(sample_loader))
        x, outputs, _ = self.term.compute(raw_batch, device=self.device)
        self.tb_logger.log_reconstruction(x, outputs["x_hat"], epoch)

        ckpt_path = self.paths.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.term.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "args": vars(self.args),
            },
            ckpt_path,
        )

        recon_path = self.paths.reconstructions_dir / f"epoch_{epoch:03d}.png"
        save_reconstruction_batch(
            x.detach().cpu(),
            outputs["x_hat"].detach().cpu(),
            recon_path,
            image_size=None,
        )

    def _select_sample_loader(self) -> torch.utils.data.DataLoader:
        """Selects a non-empty loader for visualization sampling.

        Returns:
            A non-empty dataloader, preferring validation loader.

        Raises:
            RuntimeError: If both validation and training dataloaders are empty.
        """
        if len(self.val_loader) > 0:
            return self.val_loader
        if len(self.train_loader) > 0:
            return self.train_loader
        raise RuntimeError("Both validation and training dataloaders are empty.")


def main() -> None:
    """Entrypoint for VAE training."""
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    manager = ExperimentManager(args)
    manager.run()


if __name__ == "__main__":
    main()
