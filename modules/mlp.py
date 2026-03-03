"""MLP building blocks used by VAE components."""

from __future__ import annotations

from typing import Iterable, List

import torch.nn as nn

from .activations import ActivationFactory


class MLP(nn.Module):
    """Configurable multi-layer perceptron.

    The module supports arbitrary hidden dimensions and configurable activation.
    It is designed as a reusable component for encoder/decoder definitions.

    Example:
        mlp = MLP(in_dim=784, hidden_dims=[512, 256], out_dim=128, activation="relu")
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int],
        out_dim: int,
        activation: str = "relu",
        final_activation: str = "identity",
    ) -> None:
        """Initializes the MLP.

        Args:
            in_dim: Input feature dimension.
            hidden_dims: Hidden layer dimensions.
            out_dim: Output feature dimension.
            activation: Activation for hidden layers.
            final_activation: Activation for final layer.
        """
        super().__init__()
        dims: List[int] = [in_dim, *list(hidden_dims), out_dim]
        factory = ActivationFactory()
        layers: List[nn.Module] = []

        for idx in range(len(dims) - 1):
            in_features, out_features = dims[idx], dims[idx + 1]
            layers.append(nn.Linear(in_features, out_features))
            # Final layer can use a separate activation for output control.
            act_name = activation if idx < len(dims) - 2 else final_activation
            layers.append(factory.build(act_name))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Runs a forward pass.

        Args:
            x: Input tensor with shape ``[batch, in_dim]``.

        Returns:
            Output tensor with shape ``[batch, out_dim]``.
        """
        return self.model(x)
