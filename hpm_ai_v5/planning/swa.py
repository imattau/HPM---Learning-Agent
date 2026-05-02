"""Scoring Weight Adaptation benchmark for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Mapping, Sequence

from ..agents import ScoringWeightAdaptationAgent
from ..core import CoreConfig, Pattern, PatternEngine, State


@dataclass(frozen=True, slots=True)
class SWAEpisodeResult:
    episode: int
    environment: str
    context: int
    selected_pattern: str | None
    reward: float
    weights: dict[str, float]
    trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SWAEnvironmentResult:
    environment: str
    optimal_reward: float
    learned_reward: float
    baseline_reward: float
    learned_ratio: float
    baseline_ratio: float
    learned_weights: dict[str, float]
    trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SWAResult:
    result: str
    reason: str
    environment_results: tuple[SWAEnvironmentResult, ...]
    trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SWAEnvironment:
    name: str
    contexts: tuple[int, ...]
    reward_table: dict[int, dict[str, float]]
    context_rewards: dict[int, dict[str, float]]


@dataclass(slots=True)
class ScoringWeightAdaptationBenchmark:
    """Learn scoring weights per environment from reward feedback."""

    config: CoreConfig = field(default_factory=lambda: CoreConfig(near_threshold=0.0, history_limit=8))
    train_episodes: int = 100
    evaluation_episodes: int = 40
    agent: ScoringWeightAdaptationAgent = field(default_factory=ScoringWeightAdaptationAgent)
    environments: tuple[SWAEnvironment, ...] = field(default_factory=lambda: (
        SWAEnvironment(
            name="NoisyNum",
            contexts=(0, 1),
            reward_table={
                0: {"density": 3.0, "context": 2.0, "utility": 1.0},
                1: {"density": -1.0, "context": 4.0, "utility": 1.0},
            },
            context_rewards={
                0: {"density": 0.5, "context": 0.3, "utility": 0.2},
                1: {"density": 0.1, "context": 0.7, "utility": 0.2},
            },
        ),
        SWAEnvironment(
            name="StableRepeat",
            contexts=(0, 1),
            reward_table={
                0: {"density": 4.0, "context": 1.0, "utility": 0.5},
                1: {"density": 4.0, "context": 1.0, "utility": 0.5},
            },
            context_rewards={
                0: {"density": 0.8, "context": 0.1, "utility": 0.1},
                1: {"density": 0.8, "context": 0.1, "utility": 0.1},
            },
        ),
        SWAEnvironment(
            name="SwitchContext",
            contexts=(0, 1),
            reward_table={
                0: {"density": 0.5, "context": 1.0, "utility": 4.0},
                1: {"density": 0.5, "context": 4.0, "utility": 1.0},
            },
            context_rewards={
                0: {"density": 0.1, "context": 0.2, "utility": 0.7},
                1: {"density": 0.1, "context": 0.7, "utility": 0.2},
            },
        ),
    ))

    def _context_schedule(self, environment: SWAEnvironment, length: int) -> list[int]:
        contexts = list(environment.contexts)
        return [contexts[index % len(contexts)] for index in range(length)]

    def _build_engine(self, environment: SWAEnvironment, context: int) -> PatternEngine:
        engine = PatternEngine(
            config=CoreConfig(
                canonicalization_mode="rotation_compression",
                near_threshold=0.0,
                history_limit=self.config.history_limit,
            )
        )
        context_signature = engine.store.context_signature({"env": environment.name, "context_hint": context})
        density = Pattern(name=f"{environment.name}_density", template=(1.0,), support=6, density=8.0, utility=0.2, last_error=0.05)
        density.context_memory[context_signature] = environment.context_rewards[context]["density"] * 5.0
        context_pattern = Pattern(name=f"{environment.name}_context", template=(2.0,), support=2, density=1.0, utility=0.5, last_error=0.3)
        context_pattern.context_memory[context_signature] = environment.context_rewards[context]["context"] * 5.0
        utility = Pattern(name=f"{environment.name}_utility", template=(3.0,), support=3, density=2.0, utility=4.0, last_error=0.45)
        utility.context_memory[context_signature] = environment.context_rewards[context]["utility"] * 5.0
        engine.store.add(density)
        engine.store.add(context_pattern)
        engine.store.add(utility)
        engine.current_state = State(value=0.0, context={"env": environment.name, "context_hint": context})
        engine.history = [engine.current_state]
        return engine

    def _reward(self, environment: SWAEnvironment, context: int, selected: str | None) -> float:
        if selected is None:
            return 0.0
        return environment.reward_table[context].get(selected.split("_")[-1], 0.0)

    def _fixed_weights(self) -> dict[str, float]:
        return {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25}

    def _best_candidate(self, environment: SWAEnvironment, contexts: Sequence[int]) -> tuple[dict[str, float], float]:
        candidates = [
            {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25},
            {"alpha": 0.15, "beta": 0.60, "gamma": 0.15, "delta": 0.10},
            {"alpha": 0.15, "beta": 0.15, "gamma": 0.60, "delta": 0.10},
            {"alpha": 0.15, "beta": 0.15, "gamma": 0.10, "delta": 0.60},
            {"alpha": 0.10, "beta": 0.30, "gamma": 0.50, "delta": 0.10},
            {"alpha": 0.10, "beta": 0.20, "gamma": 0.30, "delta": 0.40},
            {"alpha": 0.20, "beta": 0.10, "gamma": 0.40, "delta": 0.30},
        ]
        best_weights = candidates[0]
        best_reward = float("-inf")
        for candidate in candidates:
            reward = self._run_fixed_policy(environment, contexts, candidate)
            if reward > best_reward:
                best_reward = reward
                best_weights = candidate
        return best_weights, best_reward

    def _run_fixed_policy(self, environment: SWAEnvironment, contexts: Sequence[int], weights: Mapping[str, float]) -> float:
        total = 0.0
        for index, context in enumerate(contexts):
            engine = self._build_engine(environment, context)
            state = State(value=0.0, step=index, context={"env": environment.name, "context_hint": context})
            engine.current_state = state
            action = engine.act(goal={**self._fixed_weights(), **dict(weights)}, horizon=1)
            selected = None if action.selected_pattern is None else action.selected_pattern.name
            total += self._reward(environment, context, selected)
        return total / len(contexts) if contexts else 0.0

    def _run_learned_policy(self, environment: SWAEnvironment, contexts: Sequence[int]) -> tuple[float, dict[str, float], list[SWAEpisodeResult]]:
        learned_results: list[SWAEpisodeResult] = []
        train_contexts = list(contexts[: self.train_episodes])
        eval_contexts = list(contexts[self.train_episodes : self.train_episodes + self.evaluation_episodes])
        for index, context in enumerate(train_contexts):
            engine = self._build_engine(environment, context)
            state = State(value=0.0, step=index, context={"env": environment.name, "context_hint": context})
            decision, selected_pattern = self.agent.select(environment.name, engine, state, explore=True, goal_overrides={"utility": 0.0})
            selected_name = None if selected_pattern is None else selected_pattern.name
            reward = self._reward(environment, context, selected_name)
            self.agent.observe_reward(environment.name, selected_pattern, state, reward)
            learned_results.append(
                SWAEpisodeResult(
                    episode=index,
                    environment=environment.name,
                    context=context,
                    selected_pattern=selected_name,
                    reward=reward,
                    weights=dict(decision.weights),
                    trace={
                        "baseline": decision.reward_baseline,
                        "selected_pattern": selected_name,
                        "weights": dict(decision.weights),
                    },
                )
            )
        eval_reward = 0.0
        for index, context in enumerate(eval_contexts):
            engine = self._build_engine(environment, context)
            state = State(value=0.0, step=index, context={"env": environment.name, "context_hint": context})
            decision, selected_pattern = self.agent.select(environment.name, engine, state, explore=False, goal_overrides={"utility": 0.0})
            selected_name = None if selected_pattern is None else selected_pattern.name
            reward = self._reward(environment, context, selected_name)
            eval_reward += reward
            learned_results.append(
                SWAEpisodeResult(
                    episode=index + len(train_contexts),
                    environment=environment.name,
                    context=context,
                    selected_pattern=selected_name,
                    reward=reward,
                    weights=dict(decision.weights),
                    trace={
                        "phase": "evaluation",
                        "selected_pattern": selected_name,
                        "weights": dict(decision.weights),
                    },
                )
            )
        return (eval_reward / len(eval_contexts) if eval_contexts else 0.0), self.agent.current_weights(environment.name), learned_results

    def run(self) -> SWAResult:
        environment_results: list[SWAEnvironmentResult] = []
        trace: dict[str, Any] = {"environments": []}
        passed = True
        for environment in self.environments:
            contexts = self._context_schedule(environment, self.train_episodes + self.evaluation_episodes)
            baseline_reward = self._run_fixed_policy(environment, contexts[self.train_episodes :], self._fixed_weights())
            best_weights, optimal_reward = self._best_candidate(environment, contexts[self.train_episodes :])
            learned_reward, learned_weights, episodes = self._run_learned_policy(environment, contexts)
            learned_ratio = learned_reward / optimal_reward if optimal_reward > 0.0 else 0.0
            baseline_ratio = baseline_reward / optimal_reward if optimal_reward > 0.0 else 0.0
            environment_passed = learned_ratio >= 0.9
            passed = passed and environment_passed
            trace["environments"].append(
                {
                    "environment": environment.name,
                    "optimal_reward": optimal_reward,
                    "learned_reward": learned_reward,
                    "baseline_reward": baseline_reward,
                    "learned_ratio": learned_ratio,
                    "baseline_ratio": baseline_ratio,
                    "best_weights": best_weights,
                    "learned_weights": learned_weights,
                    "episodes": [episode.trace for episode in episodes[:6]],
                }
            )
            environment_results.append(
                SWAEnvironmentResult(
                    environment=environment.name,
                    optimal_reward=optimal_reward,
                    learned_reward=learned_reward,
                    baseline_reward=baseline_reward,
                    learned_ratio=learned_ratio,
                    baseline_ratio=baseline_ratio,
                    learned_weights=learned_weights,
                    trace={
                        "best_weights": best_weights,
                        "episodes": [episode.trace for episode in episodes[:6]],
                    },
                )
            )
        reason = "success" if passed else "scoring weight adaptation failed"
        return SWAResult(
            result="success" if passed else "failure",
            reason=reason,
            environment_results=tuple(environment_results),
            trace=trace,
        )
