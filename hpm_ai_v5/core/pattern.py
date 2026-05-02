"""Minimal pattern object for HPM v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log1p
from numbers import Real
from typing import Any, Mapping, Sequence

from .delta import Delta, _as_tuple, _is_sequence
from .state import State


def _mean_abs_error(left: Sequence[Any], right: Sequence[Any], *, scale: float = 1.0) -> float:
    limit = min(len(left), len(right))
    if not limit:
        return float(abs(len(left) - len(right))) / max(1.0, max(len(left), len(right)))
    total = 0.0
    for index in range(limit):
        total += abs(float(left[index]) - float(right[index]))
    scale = max(1.0, float(scale))
    length_penalty = abs(len(left) - len(right)) / max(1.0, max(len(left), len(right)))
    return (total / limit) / scale + length_penalty


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


def canonicalize_sequence(sequence: Sequence[Any], *, mode: str = "rotation_compression") -> tuple[Any, ...]:
    """Return a compact canonical form for a sequence."""

    if mode == "strict":
        return tuple(sequence)
    if mode == "rotation":
        return _canonical_rotation(sequence)
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

    def canonical_template(self, *, canonicalization_mode: str = "rotation_compression") -> tuple[float, ...]:
        return tuple(float(item) for item in canonicalize_sequence(self.template, mode=canonicalization_mode))

    def distance(
        self,
        observation: Any,
        *,
        canonicalization_mode: str = "rotation_compression",
        distance_scale: float = 1.0,
    ) -> float:
        candidate = _as_tuple(observation)
        if not self.template:
            return float(len(candidate))
        if all(isinstance(item, Real) for item in candidate + self.template):
            return _mean_abs_error(
                self.canonical_template(canonicalization_mode=canonicalization_mode),
                tuple(float(item) for item in canonicalize_sequence(candidate, mode=canonicalization_mode)),
                scale=distance_scale,
            )
        return 1.0 if canonicalize_sequence(candidate, mode=canonicalization_mode) != canonicalize_sequence(self.template, mode=canonicalization_mode) else 0.0

    def predict(self, state: State) -> State:
        """Predict a one-step continuation."""

        if not self.template:
            return state

        if isinstance(state.value, Real):
            index = state.step % len(self.template)
            return state.evolve(float(state.value) + self.template[index])

        if _is_sequence(state.value):
            values = tuple(float(item) for item in state.value if isinstance(item, Real))
            if len(values) == len(self.template):
                return state.evolve(tuple(values[index] + self.template[index] for index in range(len(self.template))))
            index = state.step % len(self.template)
            return state.evolve(tuple(state.value) + (self.template[index],))

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

    def _cap_context_memory(self, limit: int) -> None:
        if limit <= 0 or len(self.context_memory) <= limit:
            return
        ordered = sorted(self.context_memory.items(), key=lambda item: item[1], reverse=True)
        self.context_memory = dict(ordered[:limit])

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
        goal_utility = self.utility + float(goal.get("utility", 0.0))
        return (alpha * accuracy) + (beta * density) + (gamma * context_match) + (delta * goal_utility)

    def update(self, observation: Delta) -> None:
        """Lightweight pattern update on the delta."""

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

    def reward(self, utility_boost: float = 0.0) -> None:
        """Update intrinsic utility from a positive outcome signal."""

        self.utility += max(0.0, utility_boost)

    def decay(
        self,
        *,
        density_decay: float = 0.0,
        utility_decay: float = 0.0,
        context_decay: float = 1.0,
        context_memory_limit: int = 0,
    ) -> None:
        """Apply gentle forgetting to stale weights."""

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
