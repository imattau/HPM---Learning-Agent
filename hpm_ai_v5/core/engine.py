"""Minimal pattern engine for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .action import Action
from .config import CoreConfig
from .delta import Delta
from .pattern import Pattern
from .reasoning import ReasoningTrace
from .sequence import PatternSequence
from .state import State
from .store import MatchResult, PatternStore


@dataclass
class PatternEngine:
    """Observe deltas, reuse known patterns, and act on goals."""

    config: CoreConfig = field(default_factory=CoreConfig)
    store: PatternStore = field(default_factory=PatternStore)
    current_state: State | None = None
    history: list[State] = field(default_factory=list)
    last_match: MatchResult | None = None
    residual_memory: dict[tuple[Any, ...], int] = field(default_factory=dict)
    sequences: list[PatternSequence] = field(default_factory=list)
    pattern_trace: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        store_was_default = self.store.config == CoreConfig()
        if store_was_default:
            self.store.config = self.config
            self.store.exact_threshold = self.config.exact_threshold
            self.store.near_threshold = self.config.near_threshold
            self.store.max_patterns = self.config.max_patterns
            self.store.canonicalization_mode = self.config.canonicalization_mode
            self.store.distance_scale = self.config.distance_scale
        else:
            self.store.exact_threshold = self.store.exact_threshold if self.store.exact_threshold is not None else self.config.exact_threshold
            self.store.near_threshold = self.store.near_threshold if self.store.near_threshold is not None else self.config.near_threshold
            self.store.max_patterns = self.store.max_patterns if self.store.max_patterns is not None else self.config.max_patterns
            self.store.canonicalization_mode = self.store.canonicalization_mode or self.config.canonicalization_mode
            self.store.distance_scale = self.store.distance_scale or self.config.distance_scale

    def _derive_context(self, state: State, delta: Delta | None = None) -> dict[str, Any]:
        context = dict(state.context)
        if delta is None:
            return context
        context["delta_kind"] = "stable" if delta.magnitude == 0 else "shift"
        context["delta_shape"] = "sequence" if isinstance(delta.value, tuple) else "scalar"
        context["delta_size"] = len(delta.value) if isinstance(delta.value, tuple) else 1
        context["delta_level"] = delta.level
        return context

    def _sequence_context(self, state: State, delta: Delta | None = None) -> str:
        return self.store.context_signature(self._derive_context(state, delta))

    def _sequence_offset(self, state: State, sequence_length: int) -> int:
        if sequence_length <= 0:
            return 0
        context = state.context if isinstance(state.context, Mapping) else {}
        phase_hint = context.get("phase_hint", context.get("phase", 0))
        position = context.get("position", state.step)
        try:
            phase_value = int(phase_hint)
        except (TypeError, ValueError):
            phase_value = 0
        try:
            position_value = int(position)
        except (TypeError, ValueError):
            position_value = state.step
        return (phase_value + position_value) % sequence_length

    @staticmethod
    def _smallest_repeating_period(names: list[str]) -> tuple[str, ...] | None:
        if len(names) < 4:
            return None
        for window in range(len(names), 3, -1):
            tail = names[-window:]
            for period in range(2, window // 2 + 1):
                if window % period != 0:
                    continue
                unit = tail[:period]
                if all(tail[index] == unit[index % period] for index in range(window)):
                    return tuple(unit)
        return None

    def _record_pattern_name(self, pattern: Pattern | None, context_signature: str) -> None:
        if pattern is None:
            return
        self.pattern_trace.append(pattern.name)
        if len(self.pattern_trace) > max(4, self.config.history_limit):
            self.pattern_trace = self.pattern_trace[-max(4, self.config.history_limit) :]
        self._maybe_promote_sequence(context_signature)

    def _record_history(self, state: State) -> None:
        self.history.append(state)
        if len(self.history) > self.config.history_limit:
            self.history = self.history[-self.config.history_limit :]

    def _apply_decay(self) -> None:
        for pattern in self.store.patterns:
            pattern.decay(
                density_decay=self.config.density_decay,
                utility_decay=self.config.utility_decay,
                context_decay=max(0.0, 1.0 - self.config.density_decay),
                context_memory_limit=self.config.context_memory_limit,
            )
        for sequence in self.sequences:
            sequence.decay(
                density_decay=self.config.density_decay,
                utility_decay=self.config.utility_decay,
                context_decay=max(0.0, 1.0 - self.config.density_decay),
                context_memory_limit=self.config.context_memory_limit,
            )

    def _reward_matching_sequences(self, utility_boost: float, context_signature: str) -> None:
        if utility_boost <= 0.0:
            return
        tail = self._smallest_repeating_period(self.pattern_trace)
        if tail is None:
            return
        for sequence in self.sequences:
            if sequence.canonical_names() == tuple(tail):
                sequence.reward(utility_boost)
                sequence.reinforce(context_signature, density_boost=0.25, context_boost=0.5)

    @staticmethod
    def _trajectory_bonus(path: list[State], goal: Mapping[str, float] | None) -> float:
        if not path:
            return 0.0
        bonus = 0.1 * len(path)
        values = [state.value for state in path]
        if len({repr(value) for value in values}) > 1:
            bonus += 0.05
        if goal:
            target = goal.get("target")
            if isinstance(target, (int, float)) and isinstance(values[-1], (int, float)):
                scale = abs(float(target)) + 1.0
                closeness = max(0.0, 1.0 - abs(float(target) - float(values[-1])) / scale)
                bonus += 5.0 * closeness
        return bonus

    def _score_trajectory(self, *, base_score: float, path: list[State], goal: Mapping[str, float] | None) -> float:
        return base_score + self._trajectory_bonus(path, goal)

    def _maybe_promote_sequence(self, context_signature: str) -> None:
        tail = self._smallest_repeating_period(self.pattern_trace)
        if tail is None:
            return
        canonical_tail = tuple(tail)
        if any(sequence.canonical_names() == canonical_tail for sequence in self.sequences):
            return
        sequence = PatternSequence(pattern_names=canonical_tail, utility=0.5 * len(canonical_tail))
        sequence.observe_support()
        sequence.reinforce(context_signature, density_boost=1.0, context_boost=1.0)
        self.sequences.append(sequence)

    def observe(self, state: State) -> MatchResult | None:
        """Observe a new state and update the pattern store from the delta."""

        if self.current_state is None:
            self.current_state = state
            self._record_history(state)
            self.last_match = None
            return None

        delta = Delta.between(self.current_state.value, state.value, level="state")
        derived_context = self._derive_context(state, delta)
        context_signature = self.store.context_signature(derived_context)
        match = self.store.match(delta.value)

        if match.status == "novel" or match.pattern is None:
            learn_source = match.residual if self.store.is_repeating(match.residual) else delta.value
            pattern = self.store.learn(learn_source)
            match = MatchResult(status="novel", pattern=pattern, distance=match.distance, residual=match.residual)
            match.pattern.reinforce(context_signature, density_boost=1.0, context_boost=1.0)
        elif match.status == "near":
            if delta.value is not None:
                delta_key = tuple(delta.value) if isinstance(delta.value, tuple) else (delta.value,)
                self.residual_memory[delta_key] = self.residual_memory.get(delta_key, 0) + 1
                if self.residual_memory[delta_key] >= 2:
                    pattern = self.store.learn(delta_key, name=f"residual_{len(self.store.patterns) + 1}")
                    pattern.reinforce(context_signature, density_boost=0.5, context_boost=0.5)
                    self.residual_memory.pop(delta_key, None)
            match.pattern.observe_support()
            match.pattern.update(Delta(before=delta.before, after=delta.after, value=match.residual, magnitude=match.distance, level="pattern"))
            match.pattern.reinforce(context_signature, density_boost=0.5, context_boost=0.75)
        elif match.status == "exact":
            match.pattern.observe_support()
            match.pattern.reinforce(context_signature, density_boost=1.0, context_boost=1.0)

        utility_boost = 0.0
        if isinstance(state.goal, Mapping):
            utility_boost = float(state.goal.get("utility", 0.0))
        if utility_boost > 0.0 and match.pattern is not None:
            match.pattern.reward(utility_boost)

        self._reward_matching_sequences(utility_boost, context_signature)
        self._apply_decay()
        self.store.prune()
        self.current_state = state
        self._record_history(state)
        self.last_match = match
        self._record_pattern_name(match.pattern, context_signature)
        return match

    def retrieve(self, observation, top_k: int = 3) -> list[Pattern]:
        return self.store.top_k(observation, k=top_k)

    def select_sequence(self, goal: Mapping[str, float] | None = None) -> PatternSequence | None:
        if not self.sequences:
            return None
        goal = goal or {}
        context_signature = self._sequence_context(self.current_state or State(value=None))
        scored = sorted(self.sequences, key=lambda sequence: sequence.score(context_signature=context_signature, goal=goal), reverse=True)
        return scored[0] if scored else None

    def select(self, goal: Mapping[str, float] | None = None, top_k: int = 3) -> Pattern | None:
        """Pick the best pattern for the current state and goal."""

        if self.current_state is None:
            return None

        if len(self.history) >= 2:
            recent_delta = Delta.between(self.history[-2].value, self.history[-1].value, level="state")
            observation = recent_delta.value
        else:
            recent_delta = None
            observation = self.current_state.value

        candidates = self.store.top_k(observation, k=top_k)
        if not candidates:
            return None

        goal = goal or {}
        context_signature = self._sequence_context(self.current_state, recent_delta)
        scored = sorted(candidates, key=lambda pattern: pattern.score(context_signature=context_signature, goal=goal), reverse=True)
        matched = [pattern for pattern in scored if pattern.context_score(context_signature) > 0.0]
        if matched and len(matched) < len(scored):
            scored = sorted(matched, key=lambda pattern: pattern.score(context_signature=context_signature, goal=goal), reverse=True)
        return scored[0] if scored else None

    def act(self, goal: Mapping[str, float] | None = None, horizon: int = 1, top_k: int = 3) -> Action:
        """Select a pattern and simulate a short forecast."""

        if self.current_state is None:
            reasoning_trace = ReasoningTrace(
                observations=[],
                candidate_patterns=[],
                candidate_sequences=[],
                candidate_trajectories=[],
                rejected_candidates={},
                score_trace={},
                selected_action={"action_type": "defer", "reason": "no_state"},
                selected_trajectory=[],
                forecast=State(value=None),
                validation={"status": "deferred"},
            )
            return Action(
                action_type="defer",
                value=None,
                confidence=0.0,
                selected_pattern=None,
                selected_sequence=None,
                trace={"reason": "no_state", "reasoning_trace": reasoning_trace.to_dict()},
                forecast=State(value=None),
                reasoning_trace=reasoning_trace,
            )

        pattern = self.select(goal=goal, top_k=top_k)
        sequence = self.select_sequence(goal=goal)
        context_signature = self._sequence_context(self.current_state)
        candidates = self.store.top_k(self.current_state.value, k=top_k)
        pattern_path = [] if pattern is None else pattern.simulate(self.current_state, horizon=horizon)
        sequence_path = (
            []
            if sequence is None
            else sequence.simulate(
                self.current_state,
                horizon=horizon,
                resolver=self.store.get,
                start_offset=self._sequence_offset(self.current_state, len(sequence.pattern_names)),
            )
        )
        pattern_score = 0.0 if pattern is None else pattern.score(
            context_signature=context_signature,
            goal=goal,
        )
        sequence_score = 0.0 if sequence is None else sequence.score(
            context_signature=context_signature,
            goal=goal,
        )
        pattern_trajectory_score = self._score_trajectory(base_score=pattern_score, path=pattern_path, goal=goal)
        sequence_trajectory_score = self._score_trajectory(base_score=sequence_score, path=sequence_path, goal=goal)
        selected_score = max(pattern_trajectory_score, sequence_trajectory_score)
        confidence = max(0.0, min(1.0, selected_score / (abs(selected_score) + 1.0)))
        min_confidence = 0.0 if goal is None else goal.get("min_confidence", 0.0)
        sequence_atomic_threshold = 0.2 if goal is None else float(goal.get("sequence_atomic_threshold", 0.2))
        allow_sequence_execution = bool(goal.get("sequence_execution", False)) if goal is not None else False
        use_sequence = sequence is not None and sequence_trajectory_score >= pattern_trajectory_score
        use_sequence_macro = (
            allow_sequence_execution
            and use_sequence
            and (sequence_trajectory_score - pattern_trajectory_score) >= sequence_atomic_threshold
        )
        if use_sequence:
            path = sequence_path
            forecast = path[-1] if path else self.current_state
        else:
            path = pattern_path
            forecast = path[-1] if path else self.current_state
        action_type = "apply_delta" if (pattern is not None or use_sequence) and confidence >= min_confidence else "defer"
        if action_type == "apply_delta" and use_sequence_macro:
            action_type = "execute_sequence"
        action_value = None
        if action_type == "apply_delta":
            action_value = forecast.value
        elif action_type == "execute_sequence" and sequence is not None:
            action_value = list(sequence.pattern_names)
        candidate_pattern_traces = [
            {
                "name": candidate.name,
                "score": candidate.score(context_signature=context_signature, goal=goal),
                "support": candidate.support,
                "trajectory_score": self._score_trajectory(
                    base_score=candidate.score(context_signature=context_signature, goal=goal),
                    path=[] if candidate is None else candidate.simulate(self.current_state, horizon=horizon),
                    goal=goal,
                ),
            }
            for candidate in candidates
        ]
        candidate_sequence_traces = [
            {
                "pattern_names": list(candidate_sequence.pattern_names),
                "score": candidate_sequence.score(context_signature=context_signature, goal=goal),
                "support": candidate_sequence.support,
                "trajectory_score": self._score_trajectory(
                    base_score=candidate_sequence.score(context_signature=context_signature, goal=goal),
                    path=candidate_sequence.simulate(
                        self.current_state,
                        horizon=horizon,
                        resolver=self.store.get,
                        start_offset=self._sequence_offset(self.current_state, len(candidate_sequence.pattern_names)),
                    ),
                    goal=goal,
                ),
            }
            for candidate_sequence in self.sequences
        ]
        rejected_candidates: dict[str, str] = {}
        for candidate in candidates:
            if pattern is None or candidate.name != pattern.name:
                rejected_candidates[f"pattern:{candidate.name}"] = "lower_score"
        for candidate_sequence in self.sequences:
            if sequence is None or tuple(candidate_sequence.pattern_names) != tuple(sequence.pattern_names):
                rejected_candidates[f"sequence:{'|'.join(candidate_sequence.pattern_names)}"] = "lower_score"
        reasoning_trace = ReasoningTrace(
            observations=[state.value for state in self.history[-3:]],
            candidate_patterns=candidate_pattern_traces,
            candidate_sequences=candidate_sequence_traces,
            candidate_trajectories=[
                {
                    "kind": "pattern",
                    "name": None if pattern is None else pattern.name,
                    "score": pattern_trajectory_score,
                    "length": len(pattern_path),
                },
                {
                    "kind": "sequence",
                    "name": None if sequence is None else "|".join(sequence.pattern_names),
                    "score": sequence_trajectory_score,
                    "length": len(sequence_path),
                    "offset": None if sequence is None else self._sequence_offset(self.current_state, len(sequence.pattern_names)),
                },
            ],
            rejected_candidates=rejected_candidates,
            score_trace={
                "pattern_score": pattern_score,
                "sequence_score": sequence_score,
                "pattern_trajectory_score": pattern_trajectory_score,
                "sequence_trajectory_score": sequence_trajectory_score,
                "selected_score": selected_score,
                "confidence": confidence,
            },
            selected_action={
                "action_type": action_type,
                "value": action_value,
                "forecast_source": "sequence" if use_sequence else "pattern",
                "sequence_execution": use_sequence_macro,
                "trajectory_mode": "full" if horizon > 1 else "single_step",
                "selected_pattern": None if pattern is None else pattern.name,
                "selected_sequence": None if sequence is None else list(sequence.pattern_names),
                "sequence_offset": None if sequence is None else self._sequence_offset(self.current_state, len(sequence.pattern_names)),
            },
            selected_trajectory=path,
            forecast=forecast,
            validation={
                "status": "accepted" if action_type == "apply_delta" else "deferred",
                "min_confidence": min_confidence,
                "confidence": confidence,
                "sequence_execution": use_sequence_macro,
            },
        )
        trace = {
            "selected_pattern": None if pattern is None else pattern.name,
            "selected_sequence": None if sequence is None else list(sequence.pattern_names),
            "confidence": confidence,
            "goal": dict(goal or {}),
            "forecast_source": "sequence" if use_sequence else "pattern",
            "sequence_execution": use_sequence_macro,
            "trajectory_mode": "full" if horizon > 1 else "single_step",
            "reasoning_trace": reasoning_trace.to_dict(),
        }
        return Action(
            action_type=action_type,
            value=action_value,
            confidence=confidence,
            selected_pattern=pattern,
            selected_sequence=sequence,
            trace=trace,
            forecast=forecast,
            reasoning_trace=reasoning_trace,
        )
