"""Autoregressive rollout utilities for latent diffusion motion generation.
ar_rollout.py
Author: HuangPeixin
Last Modified: 2026-03-09

This module provides production-ready rollout components that connect:
1. AR-LDM latent sampler (predicting FSQ latent ``z_q``).
2. Frozen FSQ decoder (mapping latent + motion context to frame blocks).

The rollout follows the FSQ data contract used during training:
- decoder condition uses history only (``n`` frames);
- decoder output predicts ``current + future`` (``1 + m`` frames);
- sliding history comes from the rightmost ``n`` frames of the predicted block.
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
        if latent.ndim != 2:
            raise ValueError(f"latent must be 2D, got {tuple(latent.shape)}.")
        if latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent dim mismatch: got {latent.shape[1]}, expected {self.latent_dim}."
            )
        if cond_motion.ndim != 2:
            raise ValueError(f"cond_motion must be 2D, got {tuple(cond_motion.shape)}.")
        if cond_motion.shape[1] != self.decoder_condition_dim:
            raise ValueError(
                "cond_motion dim mismatch: "
                f"got {cond_motion.shape[1]}, expected {self.decoder_condition_dim}."
            )
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

    @torch.no_grad()
    def rollout(
        self,
        seed_history: torch.Tensor,
        text_token: torch.Tensor,
        num_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a full motion sequence with autoregressive sliding window.

        Args:
            seed_history: Initial real motion history ``[n, frame_dim]``.
            text_token: Trajectory-level text token ``[text_dim]``.
            num_steps: Number of autoregressive generation steps.

        Returns:
            Tuple:
                - ``generated_sequence``: Full generated sequence including seed history,
                  where each AR step appends generated ``current + future`` blocks.
                - ``generated_future_concat``: Concatenation of all generated
                  ``current + future`` blocks, shape ``[num_steps * (1 + m), frame_dim]``.
        """
        n = int(self.config.history_frames)
        m = int(self.config.future_frames)
        frame_dim = int(self.config.frame_dim)

        if n > (1 + m):
            raise ValueError(
                "Invalid rollout config: history_frames must satisfy n <= 1 + m. "
                f"Got n={n}, m={m}."
            )
        if seed_history.shape != (n, frame_dim):
            raise ValueError(
                f"seed_history must be {(n, frame_dim)}, got {tuple(seed_history.shape)}."
            )
        if text_token.ndim != 1:
            raise ValueError("text_token must be 1D.")
        expected_decoder_cond_dim = n * frame_dim
        if self.fsq_decoder.decoder_condition_dim != expected_decoder_cond_dim:
            raise ValueError(
                "FSQ decoder condition dim mismatch: "
                f"checkpoint expects {self.fsq_decoder.decoder_condition_dim}, "
                f"but rollout config requires {expected_decoder_cond_dim} (=n*frame_dim)."
            )
        expected_target_dim = (1 + m) * frame_dim
        if self.fsq_decoder.target_dim != expected_target_dim:
            raise ValueError(
                "FSQ decoder target dim mismatch: "
                f"checkpoint expects {self.fsq_decoder.target_dim}, "
                f"but rollout config requires {expected_target_dim} (=(1+m)*frame_dim)."
            )
        ldm_cond_dim = int(getattr(self.ldm_model.config, "cond_motion_dim", -1))
        expected_ldm_cond_dim = n * frame_dim
        if ldm_cond_dim != expected_ldm_cond_dim:
            raise ValueError(
                "LDM cond_motion dim mismatch: "
                f"model expects {ldm_cond_dim}, but rollout provides {expected_ldm_cond_dim}."
            )

        device = next(self.ldm_model.parameters()).device
        curr_history = seed_history.to(device)
        cond_text = text_token.to(device).unsqueeze(0)

        generated_blocks: list[torch.Tensor] = []
        full_sequence: list[torch.Tensor] = [curr_history.cpu()]

        for _ in range(num_steps):
            # LDM condition follows training semantics: history only.
            cond_motion = curr_history.reshape(1, -1)
            latent = self.ldm_model.sample_latent(
                cond_motion=cond_motion,
                cond_text=cond_text,
                steps=self.config.diffusion_steps,
            )
            # FSQ decoder condition is history only.
            decoder_condition = curr_history.reshape(1, -1)
            pred_flat = self.fsq_decoder.decode(
                latent=latent,
                cond_motion=decoder_condition,
            )
            pred_block = pred_flat.reshape(1 + m, frame_dim)

            generated_blocks.append(pred_block.cpu())
            # Keep the full predicted block to match requested sequence semantics.
            full_sequence.append(pred_block.cpu())

            # Next cond_motion is the rightmost n frames from previous (1+m) output.
            curr_history = pred_block[-n:].to(device)

        generated_future_concat = torch.cat(generated_blocks, dim=0)
        generated_sequence = torch.cat(full_sequence, dim=0)
        return generated_sequence, generated_future_concat
