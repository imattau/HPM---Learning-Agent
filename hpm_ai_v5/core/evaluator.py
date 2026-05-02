"""Simple evaluators for patterns and polygraphs."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Any, Mapping

from .action import Action
from .engine import PatternEngine


@dataclass(frozen=True, slots=True)
class PolygraphScore:
    """Summary of a polygraph's reliability."""

    concentration: float
    average_density: float
    fragmentation: float
    score: float


@dataclass(frozen=True, slots=True)
class PolygraphAgreement:
    """Consensus summary across multiple polygraphs."""

    selected_key: str | None
    support: int
    dispersion: int
    score: float


class PolygraphEvaluator:
    """Score a polygraph from the patterns learned within it."""

    def __init__(self, *, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0, epsilon: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def score_engine(self, engine: PatternEngine) -> PolygraphScore:
        patterns = engine.store.patterns
        if not patterns:
            return PolygraphScore(concentration=0.0, average_density=0.0, fragmentation=0.0, score=0.0)

        total_support = max(1.0, fsum(pattern.support for pattern in patterns))
        top_support = max(pattern.support for pattern in patterns)
        concentration = top_support / total_support
        average_density = fsum(pattern.density for pattern in patterns) / len(patterns)
        fragmentation = float(len(patterns))
        score = (self.alpha * concentration) + (self.beta * average_density) - (self.gamma * fragmentation)
        return PolygraphScore(
            concentration=concentration,
            average_density=average_density,
            fragmentation=fragmentation,
            score=score,
        )

    def select_view(self, scores: Mapping[str, PolygraphScore]) -> str | None:
        if not scores:
            return None
        return max(scores.items(), key=lambda item: item[1].score)[0]

    @staticmethod
    def _action_key(action: Action) -> str:
        if action.selected_pattern is not None:
            return f"pattern:{action.selected_pattern.name}"
        if action.selected_sequence is not None:
            return f"sequence:{'|'.join(action.selected_sequence.pattern_names)}"
        return f"action:{action.action_type}"

    def agreement(
        self,
        actions: Mapping[str, Action],
        scores: Mapping[str, PolygraphScore],
    ) -> PolygraphAgreement:
        if not actions:
            return PolygraphAgreement(selected_key=None, support=0, dispersion=0, score=0.0)

        votes: dict[str, float] = {}
        for view_name, action in actions.items():
            weight = max(0.0, scores.get(view_name, PolygraphScore(0.0, 0.0, 0.0, 0.0)).score) + 1.0
            key = self._action_key(action)
            votes[key] = votes.get(key, 0.0) + weight

        selected_key, support_weight = max(votes.items(), key=lambda item: item[1])
        dispersion = max(0, len(votes) - 1)
        score = support_weight - float(dispersion)
        return PolygraphAgreement(
            selected_key=selected_key,
            support=int(round(support_weight)),
            dispersion=dispersion,
            score=score,
        )
