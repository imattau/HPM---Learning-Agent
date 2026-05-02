"""Agent-side utility learning for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Mapping

from ..core import Pattern, PatternEngine, State


@dataclass(frozen=True, slots=True)
class UtilityCandidate:
    pattern: Pattern
    base_score: float
    utility_bias: float
    total_score: float


@dataclass
class UtilityDecision:
    pattern: Pattern | None
    confidence: float
    candidates: list[UtilityCandidate] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class UtilityLearningAgent:
    """Learn context-specific utility preferences over reusable patterns."""

    engine: PatternEngine
    learning_rate: float = 0.2
    exploration_rate: float = 0.2
    seed: int = 7
    utility_memory: dict[tuple[str, str], float] = field(default_factory=dict)
    rng: Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)

    def _context_signature(self, state: State) -> str:
        return self.engine.store.context_signature(state.context)

    def _utility_bias(self, pattern: Pattern, context_signature: str) -> float:
        return self.utility_memory.get((pattern.name, context_signature), 0.0)

    def select(self, state: State, *, goal: Mapping[str, float] | None = None, top_k: int = 5, explore: bool = False) -> UtilityDecision:
        goal = dict(goal or {})
        goal.setdefault("utility", 0.0)
        context_signature = self._context_signature(state)
        candidates = self.engine.store.patterns[: max(0, top_k)] or list(self.engine.store.patterns)
        if not candidates:
            return UtilityDecision(pattern=None, confidence=0.0, trace={"context_signature": context_signature, "reason": "no_candidates"})

        scored: list[UtilityCandidate] = []
        for pattern in candidates:
            base_score = pattern.score(context_signature=context_signature, goal=goal)
            utility_bias = self._utility_bias(pattern, context_signature)
            scored.append(
                UtilityCandidate(
                    pattern=pattern,
                    base_score=base_score,
                    utility_bias=utility_bias,
                    total_score=base_score + utility_bias,
                )
            )

        if explore and self.rng.random() < self.exploration_rate:
            chosen = self.rng.choice(candidates)
        else:
            chosen = max(scored, key=lambda item: item.total_score).pattern

        chosen_score = next(item.total_score for item in scored if item.pattern is chosen)
        confidence = max(0.0, min(1.0, chosen_score / (abs(chosen_score) + 1.0)))
        return UtilityDecision(
            pattern=chosen,
            confidence=confidence,
            candidates=scored,
            trace={
                "context_signature": context_signature,
                "explore": explore,
                "candidates": [
                    {
                        "name": item.pattern.name,
                        "base_score": item.base_score,
                        "utility_bias": item.utility_bias,
                        "total_score": item.total_score,
                    }
                    for item in scored
                ],
            },
        )

    def observe_reward(self, pattern: Pattern, state: State, reward: float) -> None:
        context_signature = self._context_signature(state)
        key = (pattern.name, context_signature)
        current = self.utility_memory.get(key, 0.0)
        updated = current + self.learning_rate * (float(reward) - current)
        self.utility_memory[key] = updated

    def utility(self, pattern: Pattern, state: State) -> float:
        return self._utility_bias(pattern, self._context_signature(state))
