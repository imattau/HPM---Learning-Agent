"""Postprocessor contracts for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..adapter import AdapterPacket
from ..core import Action


@dataclass(frozen=True, slots=True)
class PostprocessedOutput:
    """Structured output emitted by postprocessing."""

    output: Any
    packet: AdapterPacket | None = None
    trace: dict[str, Any] = field(default_factory=dict)


class Postprocessor(Protocol):
    """Composable output validator / renderer."""

    name: str
    requires: list[str]
    provides: list[str]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raise NotImplementedError

    def postprocess(self, action: Action, *, context: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError
