"""Learned Utility Benchmark for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Iterable

from ..agents import UtilityDecision, UtilityLearningAgent
from ..core import Pattern, PatternEngine, State


@dataclass(frozen=True, slots=True)
class LUBEpisodeResult:
    index: int
    context: int
    selected_pattern: str | None
    reward: float
    correct_choice: bool
    trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LUBResult:
    result: str
    reason: str
    preference_accuracy: float
    average_reward: float
    utility_separation: float
    episode_results: tuple[LUBEpisodeResult, ...]
    trace: dict[str, Any]


@dataclass(slots=True)
class LearnedUtilityBenchmark:
    """Learn context-conditioned utility from observed reward."""

    engine: PatternEngine = field(default_factory=PatternEngine)
    learning_rate: float = 0.2
    exploration_rate: float = 0.25
    train_episodes: int = 100
    evaluation_episodes: int = 40
    safe_name: str = "P_safe"
    risky_name: str = "Q_risky"
    agent: UtilityLearningAgent = field(init=False)

    def __post_init__(self) -> None:
        if not self.engine.store.patterns:
            self.engine.store.add(Pattern(name=self.safe_name, template=(1.0,), support=1, density=1.0))
            self.engine.store.add(Pattern(name=self.risky_name, template=(10.0,), support=1, density=1.0))
        self.agent = UtilityLearningAgent(
            engine=self.engine,
            learning_rate=self.learning_rate,
            exploration_rate=self.exploration_rate,
        )

    @staticmethod
    def _context_signature(context: int) -> dict[str, Any]:
        return {"mode": f"context_{context}", "context_hint": context}

    def _reward(self, pattern_name: str | None, context: int) -> float:
        if pattern_name == self.safe_name:
            return 1.0
        if context == 0:
            return 10.0
        return -20.0

    def _best_pattern(self, context: int) -> str:
        return self.risky_name if context == 0 else self.safe_name

    def _run_episode(self, index: int, context: int, *, train: bool, explore: bool) -> LUBEpisodeResult:
        state = State(value=0.0, context=self._context_signature(context))
        decision: UtilityDecision = self.agent.select(state, explore=explore)
        pattern = decision.pattern
        reward = self._reward(None if pattern is None else pattern.name, context)
        if train and pattern is not None:
            self.agent.observe_reward(pattern, state, reward)
        selected_name = None if pattern is None else pattern.name
        correct_choice = selected_name == self._best_pattern(context)
        return LUBEpisodeResult(
            index=index,
            context=context,
            selected_pattern=selected_name,
            reward=reward,
            correct_choice=correct_choice,
            trace={
                "selected_pattern": selected_name,
                "correct_pattern": self._best_pattern(context),
                "confidence": decision.confidence,
                "decision_trace": decision.trace,
                "utility_memory": dict(self.agent.utility_memory),
            },
        )

    def _utility_separation(self) -> float:
        context0 = self._context_signature(0)
        context1 = self._context_signature(1)
        risky0 = self.agent.utility(self.engine.store.get(self.risky_name), State(value=0.0, context=context0)) if self.engine.store.get(self.risky_name) else 0.0
        safe0 = self.agent.utility(self.engine.store.get(self.safe_name), State(value=0.0, context=context0)) if self.engine.store.get(self.safe_name) else 0.0
        risky1 = self.agent.utility(self.engine.store.get(self.risky_name), State(value=0.0, context=context1)) if self.engine.store.get(self.risky_name) else 0.0
        safe1 = self.agent.utility(self.engine.store.get(self.safe_name), State(value=0.0, context=context1)) if self.engine.store.get(self.safe_name) else 0.0
        gap0 = max(0.0, risky0 - safe0) / 10.0
        gap1 = max(0.0, safe1 - risky1) / 20.0
        return min(1.0, 0.5 * (gap0 + gap1))

    def run(self) -> LUBResult:
        training_results: list[LUBEpisodeResult] = []
        evaluation_results: list[LUBEpisodeResult] = []
        schedule = [0, 1]

        # Seed both context branches with a tiny amount of exploration.
        bootstrap: list[tuple[int, str]] = [
            (0, self.safe_name),
            (0, self.risky_name),
            (1, self.safe_name),
            (1, self.risky_name),
        ]
        for index, (context, forced_pattern) in enumerate(bootstrap):
            state = State(value=0.0, context=self._context_signature(context))
            pattern = self.engine.store.get(forced_pattern)
            reward = self._reward(forced_pattern, context)
            if pattern is not None:
                self.agent.observe_reward(pattern, state, reward)
            training_results.append(
                LUBEpisodeResult(
                    index=index,
                    context=context,
                    selected_pattern=forced_pattern,
                    reward=reward,
                    correct_choice=forced_pattern == self._best_pattern(context),
                    trace={
                        "bootstrap": True,
                        "selected_pattern": forced_pattern,
                        "utility_memory": dict(self.agent.utility_memory),
                    },
                )
            )

        for index in range(self.train_episodes):
            context = schedule[index % len(schedule)]
            result = self._run_episode(index + len(bootstrap), context, train=True, explore=True)
            training_results.append(result)

        for index in range(self.evaluation_episodes):
            context = schedule[index % len(schedule)]
            result = self._run_episode(index, context, train=False, explore=False)
            evaluation_results.append(result)

        preference_accuracy = (
            sum(1 for result in evaluation_results if result.correct_choice) / len(evaluation_results)
            if evaluation_results
            else 0.0
        )
        average_reward = fmean(result.reward for result in evaluation_results) if evaluation_results else 0.0
        utility_separation = self._utility_separation()
        normalized_reward = max(0.0, min(1.0, (average_reward + 20.0) / 30.0))
        score = fmean([preference_accuracy, normalized_reward, utility_separation])
        passed = preference_accuracy >= 0.8 and utility_separation >= 0.6
        reason = "success" if passed else "utility learning failed"
        trace = {
            "training": [result.trace for result in training_results[:6]],
            "evaluation": [result.trace for result in evaluation_results[:6]],
            "utilities": dict(self.agent.utility_memory),
        }
        return LUBResult(
            result="success" if passed else "failure",
            reason=reason,
            preference_accuracy=preference_accuracy,
            average_reward=average_reward,
            utility_separation=utility_separation,
            episode_results=tuple(evaluation_results),
            trace={
                **trace,
                "score": score,
                "normalized_reward": normalized_reward,
                "preference_accuracy": preference_accuracy,
            },
        )
