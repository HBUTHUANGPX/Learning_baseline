"""Hydra entrypoint for context-conditioned motion VQ/FSQ training."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover
    hydra = None
    DictConfig = Any  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.train_motion_vqvae import main as train_main


def _cfg_to_namespace(cfg: DictConfig) -> SimpleNamespace:
    """Converts Hydra config to training-namespace schema.

    Args:
        cfg: Composed Hydra configuration.

    Returns:
        Namespace accepted by ``train_motion_vqvae.main``.
    """
    return SimpleNamespace(
        model=str(cfg.model.name),
        embedding_dim=int(cfg.model.embedding_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_embeddings=int(cfg.model.num_embeddings),
        fsq_levels=int(cfg.model.fsq_levels),
        ifsq_boundary_fn=str(getattr(cfg.model, "ifsq_boundary_fn", "sigmoid")),
        ifsq_boundary_scale=float(getattr(cfg.model, "ifsq_boundary_scale", 1.6)),
        beta=float(getattr(cfg.model, "beta", 0.25)),
        recon_loss_mode=str(cfg.model.recon_loss_mode),
        batch_size=int(cfg.data.batch_size),
        epochs=int(cfg.train.epochs),
        lr=float(cfg.optim.lr),
        seed=int(cfg.train.seed),
        deterministic=bool(cfg.train.deterministic),
        device=str(cfg.train.device),
        motion_files=",".join(cfg.data.motion_files),
        motion_file_yaml=str(cfg.data.motion_file_yaml),
        motion_group=str(cfg.data.motion_group),
        motion_feature_keys=",".join(cfg.data.motion_feature_keys),
        motion_frame_stride=int(cfg.data.motion_frame_stride),
        motion_cache_device=str(cfg.data.motion_cache_device),
        motion_normalize=bool(cfg.data.motion_normalize),
        history_frames=int(cfg.data.history_frames),
        future_frames=int(cfg.data.future_frames),
        val_ratio=float(cfg.data.val_ratio),
        log_root=str(cfg.log.root),
    )


def _main_impl(cfg: DictConfig) -> None:
    """Runs motion training with composed Hydra configuration.

    Args:
        cfg: Composed Hydra configuration.
    """
    print(OmegaConf.to_yaml(cfg))
    args = _cfg_to_namespace(cfg)
    train_main(args)


if hydra is not None:
    main = hydra.main(version_base=None, config_path="../configs", config_name="config")(
        _main_impl
    )
else:

    def main() -> None:
        """Fallback entrypoint when hydra-core is missing."""
        raise ModuleNotFoundError(
            "hydra-core is required for scripts/train_hydra.py. "
            "Install it via requirements.txt or `pip install hydra-core`."
        )


if __name__ == "__main__":
    main()
