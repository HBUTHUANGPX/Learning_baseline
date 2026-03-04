"""Generic registry infrastructure for pluggable components."""

from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name-to-object registry with decorator-based registration.

    This helper enables plugin-style extension for datasets, models, and
    algorithms without adding large if-else blocks in training entrypoints.
    """

    def __init__(self, name: str) -> None:
        """Initializes a registry instance.

        Args:
            name: Human-readable registry name used in error messages.
        """
        self.name = name
        self._entries: Dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        """Decorator that registers an object under a key.

        Args:
            key: Unique registration key.

        Returns:
            Decorator that stores target object into registry.
        """

        def _decorator(obj: T) -> T:
            normalized = key.strip().lower()
            if normalized in self._entries:
                raise ValueError(f"Key '{key}' already registered in '{self.name}'.")
            self._entries[normalized] = obj
            return obj

        return _decorator

    def get(self, key: str) -> T:
        """Fetches a registered object by key.

        Args:
            key: Registration key.

        Returns:
            Registered object.

        Raises:
            KeyError: If key is not registered.
        """
        normalized = key.strip().lower()
        if normalized not in self._entries:
            supported = ", ".join(sorted(self._entries.keys()))
            raise KeyError(
                f"Unknown key '{key}' in registry '{self.name}'. Supported: {supported}"
            )
        return self._entries[normalized]

    def keys(self) -> list[str]:
        """Returns sorted registry keys."""
        return sorted(self._entries.keys())
