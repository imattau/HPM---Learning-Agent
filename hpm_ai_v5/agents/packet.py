"""Shared agent packet model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentPacket:
    """Mutable packet shared across agent steps."""

    raw_input: Any
    goal: Any = None
    context: dict[str, Any] = field(default_factory=dict)
    views: list[Any] = field(default_factory=list)
    core_actions: list[Any] = field(default_factory=list)
    candidate_outputs: list[Any] = field(default_factory=list)
    agent_trace: list[str] = field(default_factory=list)
    final_output: Any = None
    state: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def log(self, agent: str, detail: Any | None = None) -> None:
        self.trace.append({"agent": agent, "detail": detail})
