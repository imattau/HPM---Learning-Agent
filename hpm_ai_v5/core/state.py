"""State objects for the v5 core.

The core stays substrate-agnostic: adapters normalize input before it
reaches this layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class State:
    """Current observation plus optional goal/context."""

    value: Any
    step: int = 0
    goal: Any | None = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def evolve(self, value: Any, *, step: int | None = None, goal: Any | None = None) -> "State":
        """Return a new state with updated content."""

        return State(
            value=value,
            step=self.step + 1 if step is None else step,
            goal=self.goal if goal is None else goal,
            context=self.context,
        )
