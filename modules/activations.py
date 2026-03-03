"""Activation function abstractions."""

from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn


class ActivationFactory:
    """Factory class that builds activation layers by name.

    This class centralizes activation creation to make network blocks more
    configurable and plug-and-play. It returns newly instantiated modules, so
    each caller receives its own activation instance.

    Example:
        factory = ActivationFactory()
        layer = factory.build("relu")
    """

    _REGISTRY: Dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.2),
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
    }

    def build(self, name: str) -> nn.Module:
        """Builds an activation module from a registered name.

        Args:
            name: Activation identifier such as ``"relu"`` or ``"gelu"``.

        Returns:
            A newly created ``nn.Module`` activation layer.

        Raises:
            ValueError: If ``name`` does not exist in the registry.
        """
        key = name.lower().strip()
        if key not in self._REGISTRY:
            supported = ", ".join(sorted(self._REGISTRY))
            raise ValueError(f"Unsupported activation '{name}'. Supported: {supported}")
        return self._REGISTRY[key]()
