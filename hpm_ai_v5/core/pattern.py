"""Minimal pattern object for HPM v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log1p
from numbers import Real
from typing import Any, Mapping, Sequence

from .delta import Delta, _as_tuple, _is_sequence
from .state import State


def _mean_abs_error(left: Sequence[Any], right: Sequence[Any]) -> float:
    limit = min(len(left), len(right))
    if not limit:
        return float(abs(len(left) - len(right)))
    total = 0.0
    for index in range(limit):
        total += abs(float(left[index]) - float(right[index]))
    return total / limit + abs(len(left) - len(right))


def _canonical_rotation(sequence: Sequence[Any]) -> tuple[Any, ...]:
    values = tuple(sequence)
    if len(values) <= 1:
        return values
    rotations = [values[index:] + values[:index] for index in range(len(values))]
    return min(rotations)


def _smallest_repeat_unit(sequence: Sequence[Any]) -> tuple[Any, ...]:
    values = tuple(sequence)
    if len(values) <= 1:
        return values
    for size in range(1, len(values) // 2 + 1):
        if len(values) % size == 0 and values == values[:size] * (len(values) // size):
            return values[:size]
    return values


def canonicalize_sequence(sequence: Sequence[Any]) -> tuple[Any, ...]:
    """Return a compact canonical form for a sequence."""

    return _smallest_repeat_unit(_canonical_rotation(sequence))


@dataclass(slots=True)
class Pattern:
    """A reusable delta pattern."""

    name: str
    template: tuple[float, ...] = field(default_factory=tuple)
    support: int = 0
    density: float = 0.0
    utility: float = 0.0
    last_error: float = 0.0
    context_memory: dict[str, float] = field(default_factory=dict)

    def canonical_template(self) -> tuple[float, ...]:
        return tuple(float(item) for item in canonicalize_sequence(self.template))

    def distance(self, observation: Any) -> float:
        candidate = _as_tuple(observation)
        if not self.template:
            return float(len(candidate))
        if all(isinstance(item, Real) for item in candidate + self.template):
            return _mean_abs_error(self.canonical_template(), tuple(float(item) for item in canonicalize_sequence(candidate)))
        return 1.0 if canonicalize_sequence(candidate) != canonicalize_sequence(self.template) else 0.0

    def predict(self, state: State) -> State:
        """Predict a one-step continuation."""

        if not self.template:
            return state

        if isinstance(state.value, Real):
            return state.evolve(float(state.value) + self.template[-1])

        if _is_sequence(state.value):
            return state.evolve(tuple(state.value) + (self.template[-1],))

        return state

    def simulate(self, state: State, horizon: int = 1) -> list[State]:
        """Simulate a short continuation."""

        current = state
        path: list[State] = []
        for _ in range(max(0, horizon)):
            current = self.predict(current)
            path.append(current)
        return path

    def context_score(self, context_signature: str | None) -> float:
        if not context_signature:
            return 0.0
        return self.context_memory.get(context_signature, 0.0)

    def score(
        self,
        context_signature: str | None = None,
        goal: Mapping[str, float] | None = None,
    ) -> float:
        """Return a weighted HPM-style score."""

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

    def update(self, observation: Delta) -> None:
        """Lightweight pattern update on the delta."""

        self.support += 1
        candidate = _as_tuple(observation.value)
        if candidate and all(isinstance(item, Real) for item in candidate):
            if not self.template:
                self.template = tuple(float(item) for item in canonicalize_sequence(candidate))
            else:
                limit = min(len(self.template), len(candidate))
                merged = [
                    (self.template[index] * (self.support - 1) + float(candidate[index])) / self.support
                    for index in range(limit)
                ]
                if len(candidate) > len(self.template):
                    merged.extend(float(item) for item in candidate[limit:])
                else:
                    merged.extend(self.template[limit:])
                self.template = tuple(float(item) for item in canonicalize_sequence(merged))
        self.last_error = observation.magnitude

    def reinforce(self, context_signature: str | None = None, *, density_boost: float = 1.0, context_boost: float = 1.0) -> None:
        """Increase persistence and situational recall."""

        self.density += max(0.0, density_boost)
        if context_signature:
            self.context_memory[context_signature] = self.context_memory.get(context_signature, 0.0) + max(0.0, context_boost)
