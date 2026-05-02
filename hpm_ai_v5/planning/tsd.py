"""Triple Sequence Discovery benchmark for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import copy

from ..core import CoreConfig, PatternEngine, State


Action = float


@dataclass(frozen=True, slots=True)
class TSDStepResult:
    step: int
    mode: str
    action: Action
    reward: float
    position: float
    selected_sequence: list[str] | None
    sequence_length: int
    confidence: float
    reasoning_trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TSDResult:
    result: str
    reason: str
    baseline_reward: float
    macro_reward: float
    reward_gain: float
    discovered_sequence_length: int
    baseline_steps: tuple[TSDStepResult, ...]
    macro_steps: tuple[TSDStepResult, ...]
    trace: dict[str, Any]


@dataclass
class UnaryCounterWorld:
    """Simple counter world with a hidden triple sequence reward."""

    target_sequence: tuple[Action, Action, Action] = (1.0, 1.0, -1.0)
    position: float = 0.0
    last_actions: list[Action] = field(default_factory=list)
    total_reward: float = 0.0

    def step(self, action: Action) -> float:
        self.last_actions.append(action)
        if action > 0.0:
            self.position += 1.0
        elif action < 0.0:
            self.position = max(0.0, self.position - 1.0)
        else:
            self.position = 0.0

        reward = 0.0
        tail = tuple(self.last_actions[-3:])
        if action == 0.0:
            reward = -1.0
            self.last_actions = []
        elif tail == self.target_sequence:
            reward = 100.0
            self.last_actions = []
            self.position = 0.0

        self.total_reward += reward
        return reward


@dataclass(slots=True)
class TripleSequenceDiscoveryBenchmark:
    """Discover and reuse a length-3 sequence, then execute it as a chunk."""

    config: CoreConfig = field(default_factory=lambda: CoreConfig(near_threshold=0.0, history_limit=24))
    train_cycles: int = 8
    evaluation_cycles: int = 4
    episode_length: int = 12
    target_sequence: tuple[Action, Action, Action] = (1.0, 1.0, -1.0)
    engine: PatternEngine = field(init=False)

    def __post_init__(self) -> None:
        self.engine = PatternEngine(config=self.config)

    @staticmethod
    def _state(world: UnaryCounterWorld, step: int, *, reward: float = 0.0, action: Action | None = None) -> State:
        return State(
            value=world.position,
            step=step,
            context={
                "mode": "tsd",
                "room": world.position,
                "last_action": action,
                "reward": reward,
            },
            goal={"utility": reward},
        )

    def _action_from_forecast(self, current: float, forecast: Any) -> Action:
        if not isinstance(forecast, State):
            return 0.0
        if not isinstance(forecast.value, (int, float)):
            return 0.0
        delta = float(forecast.value) - float(current)
        if delta > 0.0:
            return 1.0
        if delta < 0.0:
            return -1.0
        return 0.0

    def _train_engine(self) -> None:
        for cycle in range(self.train_cycles):
            world = UnaryCounterWorld(target_sequence=self.target_sequence)
            self.engine.current_state = None
            self.engine.history = []
            self.engine.last_match = None
            for step in range(self.episode_length):
                current_state = self._state(world, step, action=None if not world.last_actions else world.last_actions[-1])
                if self.engine.current_state is None:
                    self.engine.current_state = current_state
                self.engine.observe(current_state)
                action = self.target_sequence[step % 3]
                reward = world.step(action)
                next_state = self._state(world, step + 1, reward=reward, action=action)
                self.engine.observe(next_state)

    def _run_policy(self, engine: PatternEngine, *, macro: bool, horizon: int) -> tuple[list[TSDStepResult], float, list[str] | None, int]:
        world = UnaryCounterWorld(target_sequence=self.target_sequence)
        steps: list[TSDStepResult] = []
        total_reward = 0.0
        locked_actions: list[Action] = []
        lock_index = 0
        discovered_sequence: list[str] | None = None

        for step in range(self.episode_length):
            current_state = self._state(world, step, action=None if not world.last_actions else world.last_actions[-1])
            engine.current_state = current_state
            goal = {
                "utility": 0.0,
                "alpha": 0.25,
                "beta": 0.25,
                "gamma": 0.25,
                "delta": 0.25,
                "min_confidence": 0.4,
                "plan_horizon": float(horizon),
                "sequence_execution": macro,
                "sequence_atomic_threshold": 0.0,
            }
            action = engine.act(goal=goal, horizon=horizon)
            selected_sequence = None if action.selected_sequence is None else list(action.selected_sequence.pattern_names)
            if discovered_sequence is None and len(selected_sequence or []) >= 3:
                discovered_sequence = list(selected_sequence or [])
            if macro and locked_actions and lock_index < len(locked_actions):
                chosen_action = locked_actions[lock_index]
            else:
                chosen_action = self._action_from_forecast(world.position, action.forecast)
                if macro and action.action_type == "execute_sequence" and action.selected_sequence is not None and len(action.selected_sequence.pattern_names) >= 3:
                    trajectory = action.reasoning_trace.selected_trajectory if action.reasoning_trace is not None else []
                    locked_actions = []
                    current_position = world.position
                    for future_state in trajectory:
                        if isinstance(future_state.value, (int, float)):
                            locked_actions.append(1.0 if float(future_state.value) > current_position else -1.0 if float(future_state.value) < current_position else 0.0)
                            current_position = float(future_state.value)
                    lock_index = 0
            reward = world.step(chosen_action)
            total_reward += reward
            next_state = self._state(world, step + 1, reward=reward, action=chosen_action)
            engine.observe(next_state)
            if macro and locked_actions:
                lock_index += 1
                if lock_index >= len(locked_actions):
                    locked_actions = []
                    lock_index = 0

            steps.append(
                TSDStepResult(
                    step=step,
                    mode="macro" if macro else "baseline",
                    action=chosen_action,
                    reward=reward,
                    position=world.position,
                    selected_sequence=selected_sequence,
                    sequence_length=0 if action.selected_sequence is None else len(action.selected_sequence.pattern_names),
                    confidence=action.confidence,
                    reasoning_trace=action.reasoning_trace.to_dict() if action.reasoning_trace is not None else {},
                )
            )
        return steps, total_reward, discovered_sequence, len(discovered_sequence or [])

    def run(self) -> TSDResult:
        self._train_engine()
        baseline_engine = copy.deepcopy(self.engine)
        macro_engine = copy.deepcopy(self.engine)
        baseline_steps, baseline_reward, _, _ = self._run_policy(baseline_engine, macro=False, horizon=1)
        macro_steps, macro_reward, discovered_sequence, discovered_length = self._run_policy(macro_engine, macro=True, horizon=3)
        reward_gain = macro_reward - baseline_reward
        passed = discovered_length >= 3 and macro_reward > baseline_reward
        reason = "success" if passed else "triple sequence discovery failed"
        trace = {
            "baseline_reward": baseline_reward,
            "macro_reward": macro_reward,
            "reward_gain": reward_gain,
            "discovered_sequence": discovered_sequence,
            "baseline": [step.reasoning_trace for step in baseline_steps[:4]],
            "macro": [step.reasoning_trace for step in macro_steps[:4]],
        }
        return TSDResult(
            result="success" if passed else "failure",
            reason=reason,
            baseline_reward=baseline_reward,
            macro_reward=macro_reward,
            reward_gain=reward_gain,
            discovered_sequence_length=discovered_length,
            baseline_steps=tuple(baseline_steps),
            macro_steps=tuple(macro_steps),
            trace=trace,
        )
