"""Shared adapter packet model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdapterPacket:
    """Mutable packet shared across adapters."""

    raw: Any
    clean: Any = None
    tokens: Any = None
    entities: Any = None
    relations: Any = None
    states: list[Any] = field(default_factory=list)
    deltas: list[Any] = field(default_factory=list)
    views: list[Any] = field(default_factory=list)
    core_action: Any = None
    draft_output: Any = None
    validated_output: Any = None
    trace: list[dict[str, Any]] = field(default_factory=list)

    def log(self, adapter: str, detail: Any | None = None) -> None:
        self.trace.append({"adapter": adapter, "detail": detail})
