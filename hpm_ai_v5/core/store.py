"""Pattern storage and retrieval for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .delta import _as_tuple
from .pattern import Pattern, canonicalize_sequence


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Minimal pattern match outcome."""

    status: str
    pattern: Pattern | None
    distance: float
    residual: tuple[Any, ...] = field(default_factory=tuple)


@dataclass
class PatternStore:
    """A small in-memory library of reusable patterns."""

    patterns: list[Pattern] = field(default_factory=list)
    exact_threshold: float = 0.0
    near_threshold: float = 1.0
    max_patterns: int = 32

    @staticmethod
    def context_signature(context: Mapping[str, Any] | None) -> str:
        if not context:
            return ""
        parts = [f"{key}={context[key]!r}" for key in sorted(context)]
        return "|".join(parts)

    def add(self, pattern: Pattern) -> Pattern:
        self.patterns.append(pattern)
        return pattern

    def top_k(self, observation: Any, k: int = 3) -> list[Pattern]:
        candidate = _as_tuple(observation)
        ranked = sorted(self.patterns, key=lambda pattern: pattern.distance(candidate))
        return ranked[: max(0, k)]

    def match(self, observation: Any) -> MatchResult:
        candidate = _as_tuple(observation)
        if not self.patterns:
            return MatchResult(status="novel", pattern=None, distance=float(len(candidate)), residual=candidate)

        best = min(self.patterns, key=lambda pattern: pattern.distance(candidate))
        distance = best.distance(candidate)
        if distance <= self.exact_threshold:
            return MatchResult(status="exact", pattern=best, distance=distance, residual=())
        if distance <= self.near_threshold:
            return MatchResult(status="near", pattern=best, distance=distance, residual=_residual(candidate, best.template))
        return MatchResult(status="novel", pattern=None, distance=distance, residual=candidate)

    def learn(self, observation: Any, name: str | None = None) -> Pattern:
        candidate = _as_tuple(observation)
        if candidate and all(isinstance(item, (int, float)) for item in candidate):
            candidate = canonicalize_sequence(candidate)
        pattern = Pattern(
            name=name or f"pattern_{len(self.patterns) + 1}",
            template=tuple(float(item) for item in candidate if isinstance(item, (int, float))),
            support=1,
            density=0.0,
        )
        return self.add(pattern)

    def is_repeating(self, observation: Any) -> bool:
        candidate = _as_tuple(observation)
        if not candidate:
            return False
        if not all(isinstance(item, (int, float)) for item in candidate):
            return False
        return canonicalize_sequence(candidate) != tuple(float(item) for item in candidate)

    def prune(self, max_patterns: int | None = None) -> int:
        limit = self.max_patterns if max_patterns is None else max_patterns
        if len(self.patterns) <= limit:
            return 0

        ordered = sorted(
            self.patterns,
            key=lambda pattern: (pattern.density + pattern.support, pattern.last_error),
        )
        to_remove = len(self.patterns) - limit
        survivors = ordered[to_remove:]
        removed = len(self.patterns) - len(survivors)
        self.patterns = survivors
        return removed


def _residual(observation: Sequence[Any], template: Sequence[Any]) -> tuple[Any, ...]:
    limit = min(len(observation), len(template))
    residual = tuple(observation[index] for index in range(limit) if observation[index] != template[index])
    if len(observation) > len(template):
        residual += tuple(observation[limit:])
    return residual
