"""Preprocessor contracts for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..adapter import AdapterPacket
from ..core import State


@dataclass(frozen=True, slots=True)
class PreprocessedInput:
    """Structured data emitted by preprocessing."""

    state: State
    context: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    packet: AdapterPacket | None = None


class Preprocessor(Protocol):
    """Composable input converter."""

    name: str
    requires: list[str]
    provides: list[str]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raise NotImplementedError

    def preprocess(self, raw: Any, *, context: dict[str, Any] | None = None) -> PreprocessedInput:
        raise NotImplementedError
