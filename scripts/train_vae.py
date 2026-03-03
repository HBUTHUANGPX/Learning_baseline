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

from modules.data import DataConfig, create_dataloader
from modules.vae import BetaVAE, ConvVAE, VanillaVAE
from modules.vqvae import FSQVAE, VQVAE
from utils import TensorboardLogger, create_experiment_paths, save_reconstruction_batch, set_seed


class ModelTerm(Protocol):
    """Interface that adapts concrete models to a unified training API."""

    model: torch.nn.Module
    metric_keys: Tuple[str, ...]
    expects_image_input: bool

    def compute(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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

    def compute(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Computes model outputs and loss dictionary.

        Args:
            x: Input tensor batch.

        Returns:
            Forward outputs and loss terms.
        """
        outputs = self.model(x)
        losses = self.model.loss_function(x, outputs)
        return outputs, losses


def _extract_inputs(batch: torch.Tensor | tuple) -> torch.Tensor:
    """Extracts input tensor from dataloader output.

    Args:
        batch: Tensor batch or tuple/list batch (e.g., ``(x, y)``).

    Returns:
        Input tensor only.
    """
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
    parser.add_argument("--dataset", type=str, default="random_binary", choices=["random_binary", "mnist"])
    parser.add_argument("--input-dim", type=int, default=784)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dims", type=str, default="256,128")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--num-embeddings", type=int, default=128)
    parser.add_argument("--fsq-levels", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
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
    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]
    if args.model == "vanilla":
        model = VanillaVAE(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            hidden_dims=hidden_dims,
            activation=args.activation,
        )
        return VAETerm(model=model, metric_keys=("loss", "recon_loss", "kl_loss"), expects_image_input=False)
    if args.model == "beta":
        model = BetaVAE(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            beta=args.beta,
            hidden_dims=hidden_dims,
            activation=args.activation,
        )
        return VAETerm(model=model, metric_keys=("loss", "recon_loss", "kl_loss"), expects_image_input=False)
    if args.model == "conv":
        return VAETerm(
            model=ConvVAE(latent_dim=args.latent_dim),
            metric_keys=("loss", "recon_loss", "kl_loss"),
            expects_image_input=True,
        )
    if args.model == "vq":
        return VAETerm(
            model=VQVAE(
                embedding_dim=args.latent_dim,
                num_embeddings=args.num_embeddings,
                beta=args.beta,
            ),
            metric_keys=("loss", "recon_loss", "quant_loss", "perplexity"),
            expects_image_input=True,
        )
    return VAETerm(
        model=FSQVAE(
            embedding_dim=args.latent_dim,
            fsq_levels=args.fsq_levels,
            beta=args.beta,
        ),
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

        data_config = DataConfig(
            dataset=args.dataset,
            input_dim=args.input_dim,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            data_root=args.data_root,
            seed=args.seed,
            flatten=not self.term.expects_image_input,
        )
        self.train_loader, self.val_loader = create_dataloader(data_config)
        self.term.model = self.term.model.to(self.device)
        self.optimizer = Adam(self.term.model.parameters(), lr=args.lr)

    def run(self) -> None:
        """Executes training loop for the configured number of epochs."""
        for epoch in range(1, self.args.epochs + 1):
            train_metrics = train_one_epoch(self.term, self.train_loader, self.optimizer, self.device)
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
            image_size=28,
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
