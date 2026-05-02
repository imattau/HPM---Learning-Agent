"""Polygraph contracts for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core import State


@dataclass(frozen=True, slots=True)
class PolygraphView:
    """One structural view of the same raw input."""

    name: str
    state: State
    context: dict[str, Any] = field(default_factory=dict)


class PolygraphGenerator(Protocol):
    """Generate multiple views from a single raw input."""

    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        raise NotImplementedError
