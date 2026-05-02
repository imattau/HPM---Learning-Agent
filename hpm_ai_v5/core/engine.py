"""Minimal pattern engine for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .action import Action
from .delta import Delta
from .pattern import Pattern
from .sequence import PatternSequence
from .state import State
from .store import MatchResult, PatternStore


@dataclass
class PatternEngine:
    """Observe deltas, reuse known patterns, and act on goals."""

    store: PatternStore = field(default_factory=PatternStore)
    current_state: State | None = None
    history: list[State] = field(default_factory=list)
    last_match: MatchResult | None = None
    residual_memory: dict[tuple[Any, ...], int] = field(default_factory=dict)
    sequences: list[PatternSequence] = field(default_factory=list)
    pattern_trace: list[str] = field(default_factory=list)

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

    def _record_pattern_name(self, pattern: Pattern | None, context_signature: str) -> None:
        if pattern is None:
            return
        self.pattern_trace.append(pattern.name)
        self._maybe_promote_sequence(context_signature)

    def _maybe_promote_sequence(self, context_signature: str) -> None:
        if len(self.pattern_trace) < 4:
            return
        tail = self.pattern_trace[-4:]
        if tail[:2] != tail[2:]:
            return
        canonical_tail = tuple(tail[:2])
        if any(sequence.canonical_names() == canonical_tail for sequence in self.sequences):
            return
        sequence = PatternSequence(pattern_names=canonical_tail)
        sequence.reinforce(context_signature, density_boost=1.0, context_boost=1.0)
        self.sequences.append(sequence)

    def observe(self, state: State) -> MatchResult | None:
        """Observe a new state and update the pattern store from the delta."""

        if self.current_state is None:
            self.current_state = state
            self.history.append(state)
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
            match.pattern.update(Delta(before=delta.before, after=delta.after, value=match.residual, magnitude=match.distance, level="pattern"))
            match.pattern.reinforce(context_signature, density_boost=0.5, context_boost=0.75)
        elif match.status == "exact":
            match.pattern.support += 1
            match.pattern.reinforce(context_signature, density_boost=1.0, context_boost=1.0)

        self.store.prune()
        self.current_state = state
        self.history.append(state)
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
        return scored[0] if scored else None

    def act(self, goal: Mapping[str, float] | None = None, horizon: int = 1, top_k: int = 3) -> Action:
        """Select a pattern and simulate a short forecast."""

        if self.current_state is None:
            return Action(action_type="defer", value=None, confidence=0.0, selected_pattern=None, selected_sequence=None, trace={"reason": "no_state"}, forecast=State(value=None))

        pattern = self.select(goal=goal, top_k=top_k)
        sequence = self.select_sequence(goal=goal)
        forecast = self.current_state if pattern is None else pattern.simulate(self.current_state, horizon=horizon)[-1]
        pattern_score = 0.0 if pattern is None else pattern.score(
            context_signature=self._sequence_context(self.current_state),
            goal=goal,
        )
        sequence_score = 0.0 if sequence is None else sequence.score(
            context_signature=self._sequence_context(self.current_state),
            goal=goal,
        )
        selected_score = max(pattern_score, sequence_score)
        confidence = max(0.0, min(1.0, selected_score / (abs(selected_score) + 1.0)))
        min_confidence = 0.0 if goal is None else goal.get("min_confidence", 0.0)
        action_type = "apply_delta" if pattern is not None and confidence >= min_confidence else "defer"
        action_value = forecast.value if action_type == "apply_delta" else None
        trace = {
            "selected_pattern": None if pattern is None else pattern.name,
            "selected_sequence": None if sequence is None else list(sequence.pattern_names),
            "confidence": confidence,
            "goal": dict(goal or {}),
        }
        return Action(
            action_type=action_type,
            value=action_value,
            confidence=confidence,
            selected_pattern=pattern,
            selected_sequence=sequence,
            trace=trace,
            forecast=forecast,
        )
