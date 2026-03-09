"""Autoregressive rollout utilities for latent diffusion motion generation.

Author: HuangPeixin
Last Modified: 2026-03-09

This module provides production-ready rollout components that connect:
1. AR-LDM latent sampler (predicting FSQ latent ``z_q``).
2. Frozen FSQ decoder (mapping latent + motion context to future frames).

The rollout strictly follows sliding-window update:
- next history frames come from the last ``n`` frames of generated future;
- next current frame is the last generated frame of this step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn

from .ar_ldm import ARLDMTransformer
from .vqvae import FrameFSQVAE


@dataclass(frozen=True)
class RolloutConfig:
    """Configuration for autoregressive rollout.

    Args:
        history_frames: Number of history frames ``n``.
        future_frames: Number of generated future frames ``m`` per step.
        frame_dim: Feature dimension per frame.
        diffusion_steps: Reverse diffusion sampling steps.
    """

    history_frames: int
    future_frames: int
    frame_dim: int
    diffusion_steps: int = 50


class FrozenFSQDecoder(nn.Module):
    """Frozen FSQ decoder loaded from training checkpoint."""

    def __init__(self, checkpoint_path: str, device: str = "auto") -> None:
        """Initializes frozen decoder.

        Args:
            checkpoint_path: Path to FSQ checkpoint.
            device: Runtime device in ``{"auto", "cpu", "cuda"}``.

        Raises:
            FileNotFoundError: If checkpoint does not exist.
            KeyError: If checkpoint is missing required keys.
        """
        super().__init__()
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"FSQ checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu")
        for key in ("encoder_input_dim", "decoder_condition_dim", "target_dim", "model_state"):
            if key not in state:
                raise KeyError(f"Checkpoint missing key: {key}")
        args = state.get("args", {})

        self.model = FrameFSQVAE(
            encoder_input_dim=int(state["encoder_input_dim"]),
            decoder_condition_dim=int(state["decoder_condition_dim"]),
            target_dim=int(state["target_dim"]),
            embedding_dim=int(args.get("embedding_dim", 32)),
            hidden_dim=int(args.get("hidden_dim", 256)),
            fsq_levels=int(args.get("fsq_levels", 8)),
            recon_loss_mode=str(args.get("recon_loss_mode", "mse")),
        )
        self.model.load_state_dict(state["model_state"], strict=True)
        self.model.eval().requires_grad_(False)

        if device == "auto":
            self.device_runtime = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable.")
            self.device_runtime = torch.device("cuda")
        elif device == "cpu":
            self.device_runtime = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")
        self.model = self.model.to(self.device_runtime)

        self.target_dim = int(state["target_dim"])
        self.decoder_condition_dim = int(state["decoder_condition_dim"])
        self.latent_dim = int(self.model.embedding_dim)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, cond_motion: torch.Tensor) -> torch.Tensor:
        """Decodes latent and condition motion to predicted frame block.

        Args:
            latent: Quantized latent tensor with shape ``[B, latent_dim]``.
            cond_motion: Condition motion tensor ``[B, decoder_condition_dim]``.

        Returns:
            Predicted frame block tensor with shape ``[B, target_dim]``.
        """
        latent_d = latent.to(self.device_runtime)
        cond_d = cond_motion.to(self.device_runtime)
        dec_in = torch.cat([latent_d, cond_d], dim=1)
        out = self.model.decoder(dec_in)
        return out.to(latent.device)


class ARLatentRolloutGenerator:
    """Autoregressive rollout generator over latent diffusion + frozen decoder."""

    def __init__(
        self,
        ldm_model: ARLDMTransformer,
        fsq_decoder: FrozenFSQDecoder,
        config: RolloutConfig,
    ) -> None:
        """Initializes rollout generator.

        Args:
            ldm_model: Trained AR-LDM denoiser.
            fsq_decoder: Frozen FSQ decoder.
            config: Rollout configuration.
        """
        self.ldm_model = ldm_model
        self.fsq_decoder = fsq_decoder
        self.config = config

    @staticmethod
    def _select_future_only(pred_frames: torch.Tensor, future_frames: int) -> torch.Tensor:
        """Selects final ``future_frames`` from decoder output block.

        Args:
            pred_frames: Decoder frame block ``[T_block, frame_dim]``.
            future_frames: Number of future frames to keep.

        Returns:
            Selected future block ``[future_frames, frame_dim]``.
        """
        if pred_frames.shape[0] <= future_frames:
            return pred_frames
        return pred_frames[-future_frames:]

    @torch.no_grad()
    def rollout(
        self,
        seed_window: torch.Tensor,
        text_token: torch.Tensor,
        num_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a full motion sequence with autoregressive sliding window.

        Args:
            seed_window: Initial window ``[n+1, frame_dim]``.
            text_token: Trajectory-level text token ``[text_dim]``.
            num_steps: Number of autoregressive generation steps.

        Returns:
            Tuple:
                - ``generated_sequence``: Full generated sequence including seed.
                - ``generated_future_concat``: Concatenation of all generated future blocks.
        """
        n = int(self.config.history_frames)
        m = int(self.config.future_frames)
        frame_dim = int(self.config.frame_dim)

        if seed_window.shape != (n + 1, frame_dim):
            raise ValueError(
                f"seed_window must be {(n + 1, frame_dim)}, got {tuple(seed_window.shape)}."
            )
        if text_token.ndim != 1:
            raise ValueError("text_token must be 1D.")

        device = next(self.ldm_model.parameters()).device
        curr_window = seed_window.to(device)
        cond_text = text_token.to(device).unsqueeze(0)

        generated_blocks: list[torch.Tensor] = []
        full_sequence: list[torch.Tensor] = [curr_window.cpu()]

        for _ in range(num_steps):
            cond_motion = curr_window.reshape(1, -1)
            latent = self.ldm_model.sample_latent(
                cond_motion=cond_motion,
                cond_text=cond_text,
                steps=self.config.diffusion_steps,
            )
            pred_flat = self.fsq_decoder.decode(latent=latent, cond_motion=cond_motion)
            pred_block = pred_flat.reshape(-1, frame_dim)
            future_block = self._select_future_only(pred_block, future_frames=m)

            generated_blocks.append(future_block.cpu())
            full_sequence.append(future_block.cpu())

            # Sliding window update: history uses last n frames from generated future.
            next_hist = future_block[-n:]
            next_curr = future_block[-1:].clone()
            curr_window = torch.cat([next_hist, next_curr], dim=0).to(device)

        generated_future_concat = torch.cat(generated_blocks, dim=0)
        generated_sequence = torch.cat(full_sequence, dim=0)
        return generated_sequence, generated_future_concat

