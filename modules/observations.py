"""Observation batching abstractions inspired by IsaacLab manager patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, MutableMapping, Sequence

import torch

BatchDict = Dict[str, object]


@dataclass(frozen=True)
class ObsTermCfg:
    """Configuration of one observation term.

    Attributes:
        name: Unique term name within one observation group.
        source_key: Key in raw sample dictionary.
        noise_std: Optional Gaussian noise standard deviation.
        clip: Optional ``(low, high)`` clipping interval.
    """

    name: str
    source_key: str
    noise_std: float = 0.0
    clip: tuple[float, float] | None = None


@dataclass(frozen=True)
class ObsGroupCfg:
    """Configuration for one observation group.

    Attributes:
        name: Group name such as ``"policy"``.
        terms: Ordered list of terms in this group.
        concatenate_terms: Whether to concatenate term tensors on the last dim.
    """

    name: str
    terms: tuple[ObsTermCfg, ...]
    concatenate_terms: bool = True


@dataclass(frozen=True)
class ObservationsCfg:
    """Top-level observation configuration.

    Attributes:
        groups: Observation groups.
        primary_group: Group used by default when consumers need one tensor.
    """

    groups: tuple[ObsGroupCfg, ...]
    primary_group: str = "policy"


class ObservationManager:
    """Builds protocol-based batch objects from raw sample dictionaries."""

    def __init__(self, cfg: ObservationsCfg) -> None:
        """Initializes manager.

        Args:
            cfg: Observation configuration object.
        """
        self.cfg = cfg
        self._group_names = {group.name for group in cfg.groups}
        if cfg.primary_group not in self._group_names:
            raise ValueError(
                f"primary_group '{cfg.primary_group}' is not defined in groups."
            )

    def build_batch(self, samples: Sequence[MutableMapping[str, object]]) -> BatchDict:
        """Builds one batch dictionary from normalized sample dictionaries.

        Args:
            samples: Sequence of raw samples.

        Returns:
            A structured batch dictionary with ``obs``, ``obs_terms``, and ``meta``.
        """
        obs: Dict[str, torch.Tensor] = {}
        obs_terms: Dict[str, Dict[str, torch.Tensor]] = {}
        meta: Dict[str, torch.Tensor] = {}

        for group in self.cfg.groups:
            term_outputs: Dict[str, torch.Tensor] = {}
            grouped_tensors: List[torch.Tensor] = []
            group_lengths: torch.Tensor | None = None
            group_mask: torch.Tensor | None = None

            for term in group.terms:
                term_tensor, lengths, mask = self._collect_term(samples, term)
                term_outputs[term.name] = term_tensor
                grouped_tensors.append(term_tensor)
                if lengths is not None:
                    group_lengths = lengths
                if mask is not None:
                    group_mask = mask

            if group.concatenate_terms and len(grouped_tensors) > 1:
                obs[group.name] = torch.cat(grouped_tensors, dim=-1)
            else:
                obs[group.name] = grouped_tensors[0]
            obs_terms[group.name] = term_outputs

            if group_lengths is not None:
                meta[f"{group.name}_lengths"] = group_lengths
            if group_mask is not None:
                meta[f"{group.name}_mask"] = group_mask

        return {"obs": obs, "obs_terms": obs_terms, "meta": meta}

    def _collect_term(
        self,
        samples: Sequence[MutableMapping[str, object]],
        term: ObsTermCfg,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Collects one term over all samples and stacks/pads it.

        Args:
            samples: Normalized sample dictionaries.
            term: One observation term config.

        Returns:
            Tuple of ``(tensor, lengths, mask)`` where lengths/mask may be None.
        """
        values = [self._to_tensor(sample[term.source_key]) for sample in samples]
        tensor, lengths, mask = _stack_or_pad(values)
        if term.noise_std > 0.0:
            tensor = tensor + torch.randn_like(tensor) * term.noise_std
        if term.clip is not None:
            low, high = term.clip
            tensor = torch.clamp(tensor, min=low, max=high)
        return tensor, lengths, mask

    @staticmethod
    def _to_tensor(value: object) -> torch.Tensor:
        """Converts scalar-like object to tensor.

        Args:
            value: Arbitrary sample value.

        Returns:
            Tensor value.
        """
        if isinstance(value, torch.Tensor):
            return value.float()
        return torch.tensor(value, dtype=torch.float32)


def _stack_or_pad(
    values: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Stacks tensors or pads variable-length tensors along first dimension.

    Args:
        values: Sequence of tensors for one term across samples.

    Returns:
        Tuple ``(batch, lengths, mask)``.
    """
    if not values:
        raise ValueError("Cannot build batch from empty tensor list.")
    reference_shape = values[0].shape
    if all(tuple(value.shape) == tuple(reference_shape) for value in values):
        return torch.stack(values, dim=0), None, None

    if values[0].ndim == 0:
        return torch.stack(values, dim=0), None, None

    trailing_shape = tuple(values[0].shape[1:])
    if any(tuple(value.shape[1:]) != trailing_shape for value in values):
        raise ValueError(
            "Variable-length padding requires matching trailing dimensions."
        )

    lengths = torch.tensor([value.shape[0] for value in values], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.zeros((len(values), max_len, *trailing_shape), dtype=values[0].dtype)
    mask = torch.zeros((len(values), max_len), dtype=torch.bool)

    for idx, value in enumerate(values):
        length = int(value.shape[0])
        padded[idx, :length] = value
        mask[idx, :length] = True
    return padded, lengths, mask
