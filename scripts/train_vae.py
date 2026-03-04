"""Training script for VAE-family models with manager-style orchestration."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Protocol, Tuple

import torch
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modules.configs import ConvVAEConfig, ImageConfig, MLPVAEConfig, VQVAEConfig
from modules.data import DataConfig, create_dataloader
from modules.vae import BetaVAE, ConvVAE, VanillaVAE
from modules.vqvae import FSQVAE, VQVAE
from utils import (
    TensorboardLogger,
    create_experiment_paths,
    save_reconstruction_batch,
    set_seed,
)


class ModelTerm(Protocol):
    """Interface that adapts concrete models to a unified training API."""

    model: torch.nn.Module
    metric_keys: Tuple[str, ...]
    expects_image_input: bool

    def compute(
        self, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Runs forward and loss computation for one batch.

        Args:
            x: Input tensor batch.

        Returns:
            Tuple of ``(outputs, losses)`` dictionaries.
        """


@dataclass
class VAETerm:
    """Default term adapter for VAE-like models with ``loss_function`` method."""

    model: torch.nn.Module
    metric_keys: Tuple[str, ...]
    expects_image_input: bool

    def compute(
        self, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Computes model outputs and loss dictionary.

        Args:
            x: Input tensor batch.

        Returns:
            Forward outputs and loss terms.
        """
        outputs = self.model(x)
        losses = self.model.loss_function(x, outputs)
        return outputs, losses


def _extract_inputs(batch: torch.Tensor | tuple | dict) -> torch.Tensor:
    """Extracts input tensor from dataloader output.

    Args:
        batch: Tensor batch, tuple/list batch, or protocol batch dictionary.

    Returns:
        Input tensor only.
    """
    if isinstance(batch, dict):
        if "obs" in batch and isinstance(batch["obs"], dict):
            obs = batch["obs"]
            if "policy" in obs:
                return obs["policy"]
            return next(iter(obs.values()))
        if "x" in batch:
            return batch["x"]
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value
        raise ValueError("Unsupported batch dictionary format.")
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _init_agg(metric_keys: Iterable[str]) -> Dict[str, float]:
    """Creates zero-valued aggregator for metric keys.

    Args:
        metric_keys: Metric names to aggregate.

    Returns:
        Mapping from metric name to zero value.
    """
    return {key: 0.0 for key in metric_keys}


def _parse_int_tuple(raw: str) -> Tuple[int, ...]:
    """Parses comma-separated integer text into tuple.

    Args:
        raw: String like ``"32,64"``.

    Returns:
        Parsed integer tuple.
    """
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def train_one_epoch(
    term: ModelTerm,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Runs one training epoch with a model term adapter.

    Args:
        term: ModelTerm adapter.
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
        x = _extract_inputs(batch).to(device)
        batch_size = int(x.shape[0])
        optimizer.zero_grad()
        _, losses = term.compute(x)
        losses["loss"].backward()
        optimizer.step()

        num_samples += batch_size
        for key in term.metric_keys:
            # Losses are per-sample averages; convert back to sample-weighted sum.
            agg[key] += float(losses[key].detach().cpu()) * batch_size

    return {key: value / max(num_samples, 1) for key, value in agg.items()}


@torch.no_grad()
def validate_one_epoch(
    term: ModelTerm,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Runs one validation epoch with a model term adapter.

    Args:
        term: ModelTerm adapter.
        loader: Validation dataloader.
        device: Computation device.

    Returns:
        Mean scalar metrics over the epoch.
    """
    term.model.eval()
    agg = _init_agg(term.metric_keys)
    num_samples = 0

    for batch in loader:
        x = _extract_inputs(batch).to(device)
        batch_size = int(x.shape[0])
        _, losses = term.compute(x)
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="random_binary",
        choices=["random_binary", "mnist", "random_sequence"],
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


def build_model_term(args: argparse.Namespace) -> VAETerm:
    """Builds a term adapter that encapsulates model and metrics contract.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Configured VAETerm instance.
    """
    image_cfg = ImageConfig(
        channels=args.image_channels,
        height=args.image_height,
        width=args.image_width,
    )
    hidden_dims = _parse_int_tuple(args.hidden_dims)
    conv_channels = _parse_int_tuple(args.conv_channels)
    vq_decoder_channels = _parse_int_tuple(args.vq_decoder_channels)
    if args.model == "vanilla":
        mlp_cfg = MLPVAEConfig(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            hidden_dims=hidden_dims,
            activation=args.activation,
            beta=args.beta,
        )
        model = VanillaVAE(
            input_dim=mlp_cfg.input_dim,
            latent_dim=mlp_cfg.latent_dim,
            config=mlp_cfg,
        )
        return VAETerm(
            model=model,
            metric_keys=("loss", "recon_loss", "kl_loss"),
            expects_image_input=False,
        )
    if args.model == "beta":
        mlp_cfg = MLPVAEConfig(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            hidden_dims=hidden_dims,
            activation=args.activation,
            beta=args.beta,
        )
        model = BetaVAE(
            input_dim=mlp_cfg.input_dim,
            latent_dim=mlp_cfg.latent_dim,
            beta=mlp_cfg.beta,
            config=mlp_cfg,
        )
        return VAETerm(
            model=model,
            metric_keys=("loss", "recon_loss", "kl_loss"),
            expects_image_input=False,
        )
    if args.model == "conv":
        conv_cfg = ConvVAEConfig(
            image=image_cfg,
            latent_dim=args.latent_dim,
            encoder_channels=conv_channels,
            bottleneck_dim=args.conv_bottleneck_dim,
        )
        return VAETerm(
            model=ConvVAE(config=conv_cfg),
            metric_keys=("loss", "recon_loss", "kl_loss"),
            expects_image_input=True,
        )
    if args.model == "vq":
        vq_cfg = VQVAEConfig(
            image=image_cfg,
            embedding_dim=args.latent_dim,
            encoder_channels=conv_channels,
            decoder_channels=vq_decoder_channels,
            num_embeddings=args.num_embeddings,
            beta=args.beta,
            fsq_levels=args.fsq_levels,
        )
        return VAETerm(
            model=VQVAE(config=vq_cfg),
            metric_keys=("loss", "recon_loss", "quant_loss", "perplexity"),
            expects_image_input=True,
        )
    vq_cfg = VQVAEConfig(
        image=image_cfg,
        embedding_dim=args.latent_dim,
        encoder_channels=conv_channels,
        decoder_channels=vq_decoder_channels,
        num_embeddings=args.num_embeddings,
        beta=args.beta,
        fsq_levels=args.fsq_levels,
    )
    return VAETerm(
        model=FSQVAE(config=vq_cfg),
        metric_keys=("loss", "recon_loss", "quant_loss", "perplexity"),
        expects_image_input=True,
    )


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
        self.term = build_model_term(args)
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
            use_batch_protocol=not args.no_batch_protocol,
        )
        self.train_loader, self.val_loader = create_dataloader(data_config)
        self.term.model = self.term.model.to(self.device)
        self.optimizer = Adam(self.term.model.parameters(), lr=args.lr)

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
        sample_batch = _extract_inputs(next(iter(sample_loader))).to(self.device)
        outputs, _ = self.term.compute(sample_batch)
        self.tb_logger.log_reconstruction(sample_batch, outputs["x_hat"], epoch)

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
            sample_batch.detach().cpu(),
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
