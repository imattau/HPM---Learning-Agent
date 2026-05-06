"""Pattern sequences for the shallow v5 hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log1p
from typing import Any, Callable, Mapping

from .state import State
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
    _cached_canon_names: tuple[str, ...] | None = field(default=None, init=False)

    def canonical_names(self) -> tuple[str, ...]:
        if self._cached_canon_names is not None:
            return self._cached_canon_names
            
        res = tuple(str(item) for item in canonicalize_sequence(self.pattern_names))
        self._cached_canon_names = res
        return res

    def context_score(self, context_signature: str | None) -> float:
        if not context_signature:
            return 0.0
        return self.context_memory.get(context_signature, 0.0)

    def _cap_context_memory(self, limit: int) -> None:
        if limit <= 0 or len(self.context_memory) <= limit:
            return
        ordered = sorted(self.context_memory.items(), key=lambda item: item[1], reverse=True)
        self.context_memory = dict(ordered[:limit])

    def score(self, *, context_signature: str | None = None, goal: Mapping[str, float] | None = None) -> float:
        goal = goal or {}
        alpha = goal.get("alpha", 1.0)
        beta = goal.get("beta", 1.0)
        gamma = goal.get("gamma", 1.0)
        delta = goal.get("delta", 1.0)
        accuracy = 1.0 / (1.0 + self.last_error)
        density = log1p(max(0.0, self.density))
        context_match = self.context_score(context_signature)
        goal_utility = self.utility + float(goal.get("utility", 0.0))
        return (alpha * accuracy) + (beta * density) + (gamma * context_match) + (delta * goal_utility)

    def reinforce(self, context_signature: str | None = None, *, density_boost: float = 1.0, context_boost: float = 1.0) -> None:
        self.density += max(0.0, density_boost)
        if context_signature:
            self.context_memory[context_signature] = self.context_memory.get(context_signature, 0.0) + max(0.0, context_boost)

    def reward(self, utility_boost: float = 0.0) -> None:
        self.utility += max(0.0, utility_boost)

    def decay(
        self,
        *,
        density_decay: float = 0.0,
        utility_decay: float = 0.0,
        context_decay: float = 1.0,
        context_memory_limit: int = 0,
    ) -> None:
        if density_decay > 0.0:
            self.density = max(0.0, self.density * (1.0 - density_decay))
        if utility_decay > 0.0:
            self.utility = max(0.0, self.utility * (1.0 - utility_decay))
        if 0.0 < context_decay < 1.0 and self.context_memory:
            self.context_memory = {key: max(0.0, value * context_decay) for key, value in self.context_memory.items()}
        self._cap_context_memory(context_memory_limit)

    def observe_support(self) -> None:
        """Record a reuse event without changing structure."""

        self.support += 1

    def simulate(
        self,
        state: State,
        horizon: int = 1,
        *,
        resolver: Callable[[str], Any | None] | None = None,
        start_offset: int = 0,
    ) -> list[State]:
        """Simulate a short continuation by replaying the pattern order."""

        if resolver is None or not self.pattern_names:
            return [state]

        current = state
        path: list[State] = []
        for index in range(max(0, horizon)):
            pattern = resolver(self.pattern_names[(start_offset + index) % len(self.pattern_names)])
            if pattern is None:
                break
            current = pattern.predict(current)
            path.append(current)
        return path
