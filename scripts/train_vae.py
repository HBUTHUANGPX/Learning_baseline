"""Training script for VAE-family models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modules.data import DataConfig, create_dataloader
from modules.vae import BetaVAE, ConvVAE, VanillaVAE
from modules.vqvae import FSQVAE, VQVAE
from utils import TensorboardLogger, create_experiment_paths, save_reconstruction_batch, set_seed


def _init_metric_container(model_name: str) -> Dict[str, float]:
    """Initializes metric accumulator based on model family.

    Args:
        model_name: Model type name.

    Returns:
        Dictionary with zero-initialized metrics.
    """
    if model_name in {"vq", "fsq"}:
        return {"loss": 0.0, "recon_loss": 0.0, "quant_loss": 0.0, "perplexity": 0.0}
    return {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}


def _extract_inputs(batch: torch.Tensor | tuple) -> torch.Tensor:
    """Extracts input tensors from dataloader batches.

    Args:
        batch: Dataloader output, either tensor or ``(input, label)`` tuple.

    Returns:
        Input tensor suitable for model forward pass.
    """
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_name: str,
) -> Dict[str, float]:
    """Runs one training epoch.

    Args:
        model: VAE-like model with ``forward`` and ``loss_function``.
        loader: Training dataloader.
        optimizer: Torch optimizer.
        device: Computation device.
        model_name: Model family identifier.

    Returns:
        Mean scalar losses over the epoch.
    """
    model.train()
    agg = _init_metric_container(model_name)
    num_batches = 0

    for batch in loader:
        x = _extract_inputs(batch).to(device)
        optimizer.zero_grad()
        outputs = model(x)
        losses = model.loss_function(x, outputs)
        losses["loss"].backward()
        optimizer.step()

        num_batches += 1
        for key in agg:
            agg[key] += float(losses[key].detach().cpu())

    return {key: value / max(num_batches, 1) for key, value in agg.items()}


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    model_name: str,
) -> Dict[str, float]:
    """Runs one validation epoch.

    Args:
        model: VAE-like model.
        loader: Validation dataloader.
        device: Computation device.
        model_name: Model family identifier.

    Returns:
        Mean scalar losses over the epoch.
    """
    model.eval()
    agg = _init_metric_container(model_name)
    num_batches = 0

    for batch in loader:
        x = _extract_inputs(batch).to(device)
        outputs = model(x)
        losses = model.loss_function(x, outputs)

        num_batches += 1
        for key in agg:
            agg[key] += float(losses[key].detach().cpu())

    return {key: value / max(num_batches, 1) for key, value in agg.items()}


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for VAE training.

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
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--log-root", type=str, default="./log")
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Builds model instance from CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Instantiated VAE model.
    """
    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]
    if args.model == "vanilla":
        return VanillaVAE(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            hidden_dims=hidden_dims,
            activation=args.activation,
        )
    if args.model == "beta":
        return BetaVAE(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            beta=args.beta,
            hidden_dims=hidden_dims,
            activation=args.activation,
        )
    if args.model == "conv":
        return ConvVAE(latent_dim=args.latent_dim)
    if args.model == "vq":
        return VQVAE(
            embedding_dim=args.latent_dim,
            num_embeddings=args.num_embeddings,
            beta=args.beta,
        )
    return FSQVAE(
        embedding_dim=args.latent_dim,
        fsq_levels=args.fsq_levels,
        beta=args.beta,
    )


def main() -> None:
    """Entrypoint for VAE training."""
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = create_experiment_paths(log_root=args.log_root)
    tb_logger = TensorboardLogger(paths.tensorboard_dir)
    image_model = args.model in {"conv", "vq", "fsq"}

    data_config = DataConfig(
        dataset=args.dataset,
        input_dim=args.input_dim,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        data_root=args.data_root,
        seed=args.seed,
        flatten=not image_model,
    )
    train_loader, val_loader = create_dataloader(data_config)

    model = build_model(args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.model)
        val_metrics = validate_one_epoch(model, val_loader, device, args.model)

        tb_logger.log_scalars(train_metrics, epoch, prefix="train")
        tb_logger.log_scalars(val_metrics, epoch, prefix="val")

        sample_batch = _extract_inputs(next(iter(val_loader))).to(device)
        sample_outputs = model(sample_batch)
        tb_logger.log_reconstruction(sample_batch, sample_outputs["x_hat"], epoch)

        ckpt_path = paths.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )

        recon_path = paths.reconstructions_dir / f"epoch_{epoch:03d}.png"
        save_reconstruction_batch(
            sample_batch.detach().cpu(),
            sample_outputs["x_hat"].detach().cpu(),
            recon_path,
            image_size=28,
        )

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"ckpt={ckpt_path.name}"
        )

    tb_logger.close()
    print(f"Training finished. Logs saved in: {paths.run_dir}")


if __name__ == "__main__":
    main()
