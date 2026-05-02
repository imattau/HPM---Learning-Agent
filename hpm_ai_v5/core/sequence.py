"""Pattern sequences for the shallow v5 hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log1p
from typing import Any, Mapping

from .pattern import canonicalize_sequence


@dataclass(slots=True)
class PatternSequence:
    """Reusable sequence of pattern names."""

    pattern_names: tuple[str, ...]
    support: int = 0
    density: float = 0.0
    utility: float = 0.0
    last_error: float = 0.0
    context_memory: dict[str, float] = field(default_factory=dict)

    def canonical_names(self) -> tuple[str, ...]:
        return tuple(str(item) for item in canonicalize_sequence(self.pattern_names))

    def context_score(self, context_signature: str | None) -> float:
        if not context_signature:
            return 0.0
        return self.context_memory.get(context_signature, 0.0)

    def score(self, *, context_signature: str | None = None, goal: Mapping[str, float] | None = None) -> float:
        goal = goal or {}
        alpha = goal.get("alpha", 1.0)
        beta = goal.get("beta", 1.0)
        gamma = goal.get("gamma", 1.0)
        delta = goal.get("delta", 1.0)
        accuracy = 1.0 / (1.0 + self.last_error)
        density = log1p(max(0.0, self.density))
        context_match = self.context_score(context_signature)
        goal_utility = goal.get("utility", self.utility)
        return (alpha * accuracy) + (beta * density) + (gamma * context_match) + (delta * goal_utility)

    def reinforce(self, context_signature: str | None = None, *, density_boost: float = 1.0, context_boost: float = 1.0) -> None:
        self.support += 1
        self.density += max(0.0, density_boost)
        if context_signature:
            self.context_memory[context_signature] = self.context_memory.get(context_signature, 0.0) + max(0.0, context_boost)

