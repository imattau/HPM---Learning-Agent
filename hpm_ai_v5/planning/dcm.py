"""Delayed Consequence Maze benchmark for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core import CoreConfig, PatternEngine, State


Action = float
WinningSequence = tuple[Action, Action, Action]


@dataclass(frozen=True, slots=True)
class DCMEpisodeResult:
    episode: int
    phase: str
    total_reward: float
    predicted_first_action: Action | None
    actual_sequence: list[Action]
    selected_sequence: list[str] | None
    sequence_length: int
    confidence: float
    deferred: bool
    reasoning_trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DCMResult:
    result: str
    reason: str
    training_reward: float
    evaluation_reward: float
    phase_results: tuple[DCMEpisodeResult, ...]
    trace: dict[str, Any]


@dataclass
class DelayedConsequenceMaze:
    """Linear maze with delayed reward after a 3-step action sequence."""

    winning_sequence: WinningSequence = (-1.0, 1.0, -1.0)
    trap_sequence: WinningSequence = (1.0, 1.0, 1.0)
    decoy_sequence: WinningSequence = (1.0, -1.0, 1.0)
    position: float = 0.0
    last_actions: list[Action] = field(default_factory=list)
    total_reward: float = 0.0

    def step(self, action: Action) -> tuple[float, float]:
        self.last_actions.append(action)
        self.position += action
        reward = 0.0
        tail = tuple(self.last_actions[-3:])
        if tail == self.winning_sequence:
            reward = 10.0
        elif tail == self.trap_sequence:
            reward = -5.0
        self.total_reward += reward
        return self.position, reward


class DelayedConsequenceMazeBenchmark:
    """Train on delayed reward and check phase-aware strategy reuse."""

    def __init__(self, engine: PatternEngine | None = None) -> None:
        defaults = DelayedConsequenceMaze()
        self.winning_sequence = defaults.winning_sequence
        self.trap_sequence = defaults.trap_sequence
        self.decoy_sequence = defaults.decoy_sequence
        self.evaluation_prefix_length = 2
        self.engine = engine or PatternEngine(
            config=CoreConfig(
                near_threshold=0.0,
                history_limit=8,
                density_decay=0.02,
                utility_decay=0.01,
            )
        )

    def _state(self, maze: DelayedConsequenceMaze, step: int, *, reward: float = 0.0, last_action: Action | None = None) -> State:
        return State(
            value=maze.position,
            step=step,
            context={
                "mode": "dcm",
                "room_id": 1,
                "last_action": last_action,
                "reward": reward,
            },
            goal={"utility": reward},
        )

    def _predicted_action(self, current_position: float, action: Any) -> Action | None:
        selected_trajectory = []
        reasoning_trace = getattr(action, "reasoning_trace", None)
        if reasoning_trace is not None:
            selected_trajectory = getattr(reasoning_trace, "selected_trajectory", [])
        if not selected_trajectory:
            forecast = getattr(action, "forecast", None)
            if forecast is not None:
                selected_trajectory = [forecast]
        if not selected_trajectory:
            return None
        first = selected_trajectory[0].value
        if not isinstance(first, (int, float)):
            return None
        if first > current_position:
            return 1.0
        if first < current_position:
            return -1.0
        return 0.0

    def _run_episode(
        self,
        episode: int,
        action_sequence: Sequence[Action],
        phase: str,
        *,
        controlled_prefix_length: int | None = None,
    ) -> DCMEpisodeResult:
        maze = DelayedConsequenceMaze()
        total_reward = 0.0
        predicted_first_action: Action | None = None
        deferred = False
        action_trace: list[Action] = []

        for step_index, intended_action in enumerate(action_sequence):
            current_state = self._state(maze, step_index, last_action=None if not action_trace else action_trace[-1])
            if step_index == 0 and not self.engine.history:
                self.engine.history = [current_state]
            self.engine.current_state = current_state
            goal = {
                "utility": 0.0,
                "alpha": 0.3,
                "beta": 0.2,
                "gamma": 0.2,
                "delta": 0.3,
                "min_confidence": 0.35,
                "plan_horizon": 3.0,
            }
            action = self.engine.act(goal=goal, horizon=3)
            if predicted_first_action is None and (controlled_prefix_length is None or step_index >= controlled_prefix_length):
                predicted_first_action = self._predicted_action(current_state.value, action)
                deferred = action.action_type == "defer"

            use_policy = controlled_prefix_length is not None and step_index >= controlled_prefix_length
            if use_policy:
                actual_action = predicted_first_action if predicted_first_action is not None else 0.0
            else:
                actual_action = float(intended_action)
            action_trace.append(actual_action)
            _, reward = maze.step(actual_action)
            total_reward += reward
            next_state = self._state(maze, step_index + 1, reward=reward, last_action=actual_action)
            self.engine.observe(next_state)

        self.engine.current_state = State(value=0.0, step=0, context={"mode": "dcm", "room_id": 1, "last_action": None, "reward": 0.0})
        self.engine.history = [self.engine.current_state]
        self.engine.last_match = None

        selected = self.engine.act(goal={"utility": 10.0, "alpha": 0.3, "beta": 0.2, "gamma": 0.2, "delta": 0.3, "min_confidence": 0.35, "plan_horizon": 3.0}, horizon=3)
        selected_sequence = None if selected.selected_sequence is None else list(selected.selected_sequence.pattern_names)
        sequence_length = 0 if selected.selected_sequence is None else len(selected.selected_sequence.pattern_names)
        evaluation_reward = 0.0
        selected_trajectory = [] if selected.reasoning_trace is None else selected.reasoning_trace.selected_trajectory
        if selected_trajectory:
            first = selected_trajectory[0].value
            if isinstance(first, (int, float)):
                predicted_action = 1.0 if first > 0.0 else -1.0 if first < 0.0 else 0.0
                if predicted_action == self.winning_sequence[0]:
                    evaluation_reward = 10.0

        return DCMEpisodeResult(
            episode=episode,
            phase=phase,
            total_reward=total_reward,
            predicted_first_action=predicted_first_action,
            actual_sequence=list(action_sequence),
            selected_sequence=selected_sequence,
            sequence_length=sequence_length,
            confidence=selected.confidence,
            deferred=deferred,
            reasoning_trace=selected.reasoning_trace.to_dict() if selected.reasoning_trace is not None else {},
        )

    def run(self) -> DCMResult:
        curriculum = (
            ("train_winning", self.winning_sequence),
            ("train_decoy", self.decoy_sequence),
            ("train_trap", self.trap_sequence),
            ("train_winning_2", self.winning_sequence),
            ("train_winning_3", self.winning_sequence),
        )
        phase_results = tuple(self._run_episode(index, sequence, phase) for index, (phase, sequence) in enumerate(curriculum))
        training_reward = sum(result.total_reward for result in phase_results)
        evaluation = self._run_episode(
            len(phase_results),
            self.winning_sequence,
            "evaluation",
            controlled_prefix_length=self.evaluation_prefix_length,
        )
        evaluation_reward = evaluation.total_reward
        first_action_correct = evaluation.predicted_first_action == self.winning_sequence[self.evaluation_prefix_length]
        sequence_discovered = evaluation.sequence_length == len(self.winning_sequence)
        rewarded_sequence = any(
            sequence and len(sequence) == len(self.winning_sequence)
            for sequence in (evaluation.selected_sequence, *[result.selected_sequence for result in phase_results])
        )
        passed = first_action_correct and sequence_discovered and training_reward > 0.0 and evaluation_reward > 0.0 and rewarded_sequence
        reason = "success" if passed else "delayed reward or sequence reuse failed"
        trace = {
            "training_reward": training_reward,
            "evaluation_reward": evaluation_reward,
            "curriculum": [phase for phase, _ in curriculum],
            "evaluation": evaluation.reasoning_trace,
        }
        return DCMResult(
            result="success" if passed else "failure",
            reason=reason,
            training_reward=training_reward,
            evaluation_reward=evaluation_reward,
            phase_results=phase_results + (evaluation,),
            trace=trace,
        )
