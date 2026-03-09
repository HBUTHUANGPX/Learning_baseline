"""Autoregressive Latent Diffusion Model (AR-LDM) core network.

Author: HuangPeixin
Last Modified: 2026-03-09

This module implements a production-ready latent diffusion denoiser for motion
generation conditioned by:

1. Motion context (history + current flattened frames).
2. Trajectory-level CLIP text token (pooler_output).

The model predicts latent noise in FSQ latent space and supports:
- pluggable denoiser backbone: ``mlp`` or ``transformer``;
- pluggable fusion strategy: ``concat`` or ``cross_attention``;
- DDPM-style training and ancestral sampling;
- optimizer/scheduler factory for training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ARLDMConfig:
    """Configuration for AR-LDM denoiser.

    Args:
        latent_dim: Dimension of FSQ latent target.
        cond_motion_dim: Flattened motion condition dimension.
        text_dim: CLIP text token dimension.
        model_dim: Internal hidden dimension.
        num_layers: Number of transformer/MLP blocks.
        num_heads: Number of attention heads for attention modules.
        dropout: Dropout probability.
        timesteps: Number of diffusion timesteps.
        beta_start: Diffusion beta schedule start.
        beta_end: Diffusion beta schedule end.
        backbone_type: Backbone type in ``{"mlp", "transformer"}``.
        fusion_type: Fusion type in ``{"concat", "cross_attention"}``.
        lr: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
    """

    latent_dim: int
    cond_motion_dim: int
    text_dim: int
    model_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    backbone_type: str = "transformer"
    fusion_type: str = "cross_attention"
    lr: float = 1e-4
    weight_decay: float = 1e-4


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding module for diffusion steps."""

    def __init__(self, dim: int) -> None:
        """Initializes sinusoidal embedding projection.

        Args:
            dim: Output embedding dimension.
        """
        super().__init__()
        self.dim = int(dim)
        self.proj = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embeds integer diffusion timesteps.

        Args:
            timesteps: Timestep tensor with shape ``[B]`` and integer values.

        Returns:
            Time embedding tensor with shape ``[B, dim]``.
        """
        half_dim = self.dim // 2
        device = timesteps.device
        dtype = torch.float32
        positions = timesteps.to(dtype=dtype).unsqueeze(1)
        scales = torch.exp(
            torch.arange(half_dim, device=device, dtype=dtype)
            * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / max(half_dim - 1, 1))
        )
        args = positions * scales.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.proj(emb)


class ARLDMTransformer(nn.Module):
    """Autoregressive latent diffusion denoiser with configurable fusion/backbone.

    The network predicts noise for latent denoising:

    ``eps_theta = f(x_t, t, cond_motion, cond_text)``
    """

    def __init__(self, config: ARLDMConfig) -> None:
        """Initializes AR-LDM model.

        Args:
            config: AR-LDM configuration dataclass.

        Raises:
            ValueError: If unsupported backbone/fusion options are provided.
        """
        super().__init__()
        self.config = config

        backbone = config.backbone_type.lower().strip()
        fusion = config.fusion_type.lower().strip()
        if backbone not in {"mlp", "transformer"}:
            raise ValueError(f"Unsupported backbone_type: {config.backbone_type}")
        if fusion not in {"concat", "cross_attention"}:
            raise ValueError(f"Unsupported fusion_type: {config.fusion_type}")
        self.backbone_type = backbone
        self.fusion_type = fusion

        self.latent_proj = nn.Linear(config.latent_dim, config.model_dim)
        self.motion_proj = nn.Linear(config.cond_motion_dim, config.model_dim)
        self.text_proj = nn.Linear(config.text_dim, config.model_dim)
        self.time_embed = SinusoidalTimeEmbedding(config.model_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.model_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        if self.backbone_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.num_heads,
                dim_feedforward=config.model_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=config.num_layers,
            )
            self.concat_head = nn.Sequential(
                nn.Linear(config.model_dim * 3, config.model_dim),
                nn.GELU(),
                nn.Linear(config.model_dim, config.latent_dim),
            )
        else:
            mlp_in = config.model_dim * (3 if self.fusion_type == "concat" else 1)
            layers: list[nn.Module] = []
            hidden = config.model_dim
            for _ in range(config.num_layers):
                layers.append(nn.Linear(mlp_in if not layers else hidden, hidden))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(config.dropout))
            layers.append(nn.Linear(hidden, config.latent_dim))
            self.mlp = nn.Sequential(*layers)

        self.out_proj = nn.Linear(config.model_dim, config.latent_dim)
        self._init_diffusion_buffers()

    def _init_diffusion_buffers(self) -> None:
        """Initializes diffusion schedule buffers."""
        cfg = self.config
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        cond_motion: torch.Tensor,
        cond_text: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts latent noise from noisy latent and conditions.

        Args:
            x_t: Noisy latent tensor ``[B, latent_dim]``.
            timesteps: Diffusion timestep tensor ``[B]``.
            cond_motion: Motion condition tensor ``[B, cond_motion_dim]``.
            cond_text: Text condition tensor ``[B, text_dim]``.

        Returns:
            Predicted noise tensor ``[B, latent_dim]``.
        """
        x_token = self.latent_proj(x_t) + self.time_embed(timesteps)
        motion_token = self.motion_proj(cond_motion)
        text_token = self.text_proj(cond_text)

        if self.fusion_type == "cross_attention":
            query = x_token.unsqueeze(1)
            context = torch.stack([motion_token, text_token], dim=1)
            attended, _ = self.cross_attn(query=query, key=context, value=context, need_weights=False)
            x_token = attended.squeeze(1)

        if self.backbone_type == "mlp":
            if self.fusion_type == "concat":
                fused = torch.cat([x_token, motion_token, text_token], dim=1)
            else:
                fused = x_token
            return self.mlp(fused)

        if self.fusion_type == "concat":
            # Transformer over concatenated token sequence.
            tokens = torch.stack([x_token, motion_token, text_token], dim=1)
            tokens = self.transformer(tokens)
            flat = tokens.reshape(tokens.shape[0], -1)
            return self.concat_head(flat)

        # Cross-attended x token goes through transformer with condition tokens.
        tokens = torch.stack([x_token, motion_token, text_token], dim=1)
        tokens = self.transformer(tokens)
        return self.out_proj(tokens[:, 0, :])

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Applies forward diffusion ``q(x_t | x_0)``.

        Args:
            x0: Clean latent tensor ``[B, latent_dim]``.
            t: Diffusion timesteps ``[B]``.
            noise: Gaussian noise tensor matching ``x0``.

        Returns:
            Noisy latent tensor ``x_t``.
        """
        sqrt_alpha = self.sqrt_alpha_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def p_losses(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Computes DDPM training loss for one batch.

        Args:
            batch: Batch dictionary containing:
                ``target_latent``, ``cond_motion``, ``cond_text``.

        Returns:
            Dictionary containing scalar loss terms.
        """
        x0 = batch["target_latent"]
        cond_motion = batch["cond_motion"]
        cond_text = batch["cond_text"]
        bsz = x0.shape[0]
        t = torch.randint(0, self.config.timesteps, (bsz,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0=x0, t=t, noise=noise)
        pred_noise = self.forward(
            x_t=x_t,
            timesteps=t,
            cond_motion=cond_motion,
            cond_text=cond_text,
        )
        loss = F.mse_loss(pred_noise, noise)
        return {"loss": loss, "diffusion_loss": loss}

    def training_step(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Runs one training step.

        Args:
            batch: Input batch dictionary.

        Returns:
            Loss dictionary.
        """
        return self.p_losses(batch)

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Builds optimizer and scheduler pair.

        Returns:
            Tuple of ``(optimizer, scheduler)``.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=100,
        )
        return optimizer, scheduler

    @torch.no_grad()
    def sample_latent(
        self,
        cond_motion: torch.Tensor,
        cond_text: torch.Tensor,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Samples clean latent from noise via reverse diffusion.

        Args:
            cond_motion: Motion condition tensor ``[B, cond_motion_dim]``.
            cond_text: Text condition tensor ``[B, text_dim]``.
            steps: Optional reduced number of reverse steps.

        Returns:
            Sampled latent tensor ``[B, latent_dim]``.
        """
        bsz = cond_motion.shape[0]
        device = cond_motion.device
        x = torch.randn((bsz, self.config.latent_dim), device=device, dtype=cond_motion.dtype)

        max_t = int(self.config.timesteps - 1)
        use_steps = int(steps) if steps is not None else int(self.config.timesteps)
        use_steps = max(1, min(use_steps, self.config.timesteps))
        time_indices = torch.linspace(max_t, 0, steps=use_steps, device=device).long()

        for t_scalar in time_indices:
            t = torch.full((bsz,), int(t_scalar.item()), device=device, dtype=torch.long)
            pred_noise = self.forward(x_t=x, timesteps=t, cond_motion=cond_motion, cond_text=cond_text)

            beta_t = self.betas[t].unsqueeze(-1)
            sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1)
            sqrt_recip_alpha = self.sqrt_recip_alphas[t].unsqueeze(-1)

            model_mean = sqrt_recip_alpha * (x - beta_t * pred_noise / sqrt_one_minus)
            if int(t_scalar.item()) > 0:
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(beta_t) * noise
            else:
                x = model_mean
        return x

