"""Hydra-based training entrypoint with layered configuration composition."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - handled explicitly at runtime.
    hydra = None
    DictConfig = Any  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.train_vae import ExperimentManager
from utils import set_seed


def _to_csv(values: Iterable[int]) -> str:
    """Converts a sequence of integers into comma-separated text."""
    return ",".join(str(int(v)) for v in values)


def _cfg_to_namespace(cfg: DictConfig) -> SimpleNamespace:
    """Converts Hydra dict config into legacy manager namespace.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        Namespace compatible with ``ExperimentManager``.
    """
    return SimpleNamespace(
        algorithm=cfg.algo.name,
        model=cfg.model.name,
        dataset=cfg.data.name,
        input_dim=int(cfg.model.input_dim),
        image_channels=int(cfg.model.image_channels),
        image_height=int(cfg.model.image_height),
        image_width=int(cfg.model.image_width),
        latent_dim=int(cfg.model.latent_dim),
        hidden_dims=_to_csv(cfg.model.hidden_dims),
        conv_channels=_to_csv(cfg.model.conv_channels),
        conv_bottleneck_dim=int(cfg.model.conv_bottleneck_dim),
        vq_decoder_channels=_to_csv(cfg.model.vq_decoder_channels),
        activation=str(cfg.model.activation),
        beta=float(cfg.model.beta),
        num_embeddings=int(cfg.model.num_embeddings),
        fsq_levels=int(cfg.model.fsq_levels),
        epochs=int(cfg.train.epochs),
        batch_size=int(cfg.data.batch_size),
        num_samples=int(cfg.data.num_samples),
        sequence_length=int(cfg.data.sequence_length),
        sequence_feature_dim=int(cfg.data.sequence_feature_dim),
        sequence_variable_length=bool(cfg.data.sequence_variable_length),
        sequence_min_length=int(cfg.data.sequence_min_length),
        no_batch_protocol=not bool(cfg.data.use_batch_protocol),
        lr=float(cfg.optim.lr),
        seed=int(cfg.train.seed),
        deterministic=bool(cfg.train.deterministic),
        device=str(cfg.train.device),
        data_root=str(cfg.data.data_root),
        log_root=str(cfg.log.root),
    )


def _main_impl(cfg: DictConfig) -> None:
    """Hydra entrypoint for layered training configuration.

    Args:
        cfg: Composed Hydra config.
    """
    print(OmegaConf.to_yaml(cfg))
    args = _cfg_to_namespace(cfg)
    set_seed(args.seed, deterministic=args.deterministic)
    manager = ExperimentManager(args)
    manager.run()


if hydra is not None:
    main = hydra.main(version_base=None, config_path="../configs", config_name="config")(
        _main_impl
    )
else:
    def main() -> None:
        """Fallback entrypoint when hydra-core is not installed."""
        raise ModuleNotFoundError(
            "hydra-core is required for scripts/train_hydra.py. "
            "Install it via requirements.txt or `pip install hydra-core`."
        )


if __name__ == "__main__":
    main()
