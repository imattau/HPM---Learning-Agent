"""Meta-learning of scoring weights for the v5 core."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log1p
from random import Random
from typing import Any, Mapping

from ..core import Action, Pattern, PatternEngine, State


@dataclass(frozen=True, slots=True)
class WeightDecision:
    weights: dict[str, float]
    action: Action
    reward_baseline: float
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoringWeightAdaptationAgent:
    """Learn a simplex of scoring weights per environment from reward feedback."""

    learning_rate: float = 0.12
    exploration_rate: float = 0.15
    baseline_rate: float = 0.05
    seed: int = 11
    weights_by_env: dict[str, dict[str, float]] = field(default_factory=dict)
    reward_baseline_by_env: dict[str, float] = field(default_factory=dict)
    rng: Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)

    @staticmethod
    def _uniform_weights() -> dict[str, float]:
        return {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25}

    @staticmethod
    def _project_simplex(weights: Mapping[str, float]) -> dict[str, float]:
        clipped = {key: max(0.0, float(value)) for key, value in weights.items()}
        total = sum(clipped.values())
        if total <= 0.0:
            return ScoringWeightAdaptationAgent._uniform_weights()
        return {key: value / total for key, value in clipped.items()}

    def _weights_for_env(self, env_name: str) -> dict[str, float]:
        if env_name not in self.weights_by_env:
            self.weights_by_env[env_name] = self._uniform_weights()
        return dict(self.weights_by_env[env_name])

    @staticmethod
    def _context_signature(context: Mapping[str, Any] | None) -> str:
        if not context:
            return ""
        return "|".join(f"{key}={context[key]!r}" for key in sorted(context))

    def _set_weights_for_env(self, env_name: str, weights: Mapping[str, float]) -> None:
        self.weights_by_env[env_name] = self._project_simplex(weights)

    def _perturb(self, weights: Mapping[str, float]) -> dict[str, float]:
        perturbed = {
            key: max(0.0, float(value) + self.rng.uniform(-0.1, 0.1))
            for key, value in weights.items()
        }
        return self._project_simplex(perturbed)

    @staticmethod
    def _trait_vector(pattern: Pattern, state: State) -> dict[str, float]:
        context_signature = ScoringWeightAdaptationAgent._context_signature(state.context)
        accuracy = 1.0 / (1.0 + max(0.0, float(pattern.last_error)))
        density = log1p(max(0.0, pattern.density)) / log1p(10.0)
        context_match = min(1.0, max(0.0, pattern.context_score(context_signature) / 5.0))
        utility = min(1.0, max(0.0, pattern.utility / 5.0))
        return {"alpha": accuracy, "beta": density, "gamma": context_match, "delta": utility}

    def select(
        self,
        env_name: str,
        engine: PatternEngine,
        state: State,
        *,
        explore: bool = False,
        top_k: int = 5,
        goal_overrides: Mapping[str, float] | None = None,
    ) -> tuple[WeightDecision, Pattern | None]:
        weights = self._weights_for_env(env_name)
        if explore and self.rng.random() < self.exploration_rate:
            weights = self._perturb(weights)
        goal = {"utility": 0.0, **weights, **dict(goal_overrides or {})}
        engine.current_state = state
        action = engine.act(goal=goal, horizon=int(goal.get("plan_horizon", 1.0)), top_k=top_k)
        baseline = self.reward_baseline_by_env.get(env_name, 0.0)
        return WeightDecision(weights=weights, action=action, reward_baseline=baseline, trace={"env": env_name, "weights": dict(weights)}), action.selected_pattern

    def observe_reward(self, env_name: str, pattern: Pattern | None, state: State, reward: float) -> None:
        baseline = self.reward_baseline_by_env.get(env_name, reward)
        advantage = float(reward) - baseline
        self.reward_baseline_by_env[env_name] = baseline + self.baseline_rate * advantage
        if pattern is None:
            return
        weights = self._weights_for_env(env_name)
        traits = self._trait_vector(pattern, state)
        updated = {}
        for key, value in weights.items():
            updated[key] = value * exp(self.learning_rate * advantage * traits[key])
        self._set_weights_for_env(env_name, updated)

    def current_weights(self, env_name: str) -> dict[str, float]:
        return self._weights_for_env(env_name)
