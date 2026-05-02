"""Structured reasoning trace for v5 core decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .state import State


@dataclass(frozen=True, slots=True)
class ReasoningTrace:
    """Inspectable record of why the core selected an action."""

    observations: list[Any] = field(default_factory=list)
    candidate_patterns: list[dict[str, Any]] = field(default_factory=list)
    candidate_sequences: list[dict[str, Any]] = field(default_factory=list)
    candidate_trajectories: list[dict[str, Any]] = field(default_factory=list)
    rejected_candidates: dict[str, str] = field(default_factory=dict)
    score_trace: dict[str, float] = field(default_factory=dict)
    selected_action: dict[str, Any] = field(default_factory=dict)
    selected_trajectory: list[State] = field(default_factory=list)
    forecast: State | None = None
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": list(self.observations),
            "candidate_patterns": [dict(item) for item in self.candidate_patterns],
            "candidate_sequences": [dict(item) for item in self.candidate_sequences],
            "candidate_trajectories": [dict(item) for item in self.candidate_trajectories],
            "rejected_candidates": dict(self.rejected_candidates),
            "score_trace": dict(self.score_trace),
            "selected_action": dict(self.selected_action),
            "selected_trajectory": [
                {"value": state.value, "context": dict(state.context), "step": state.step, "goal": state.goal}
                for state in self.selected_trajectory
            ],
            "forecast": None if self.forecast is None else {"value": self.forecast.value, "context": dict(self.forecast.context)},
            "validation": dict(self.validation),
        }
