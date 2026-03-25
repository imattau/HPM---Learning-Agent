from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hpm.completion_types import (
    DecisionTrace,
    EvaluatorVector,
    MetaEvaluatorState,
    FieldConstraint,
    LifecycleSummary,
    PatternIdentity,
    PatternState,
)


class EvaluatorArbitrator:
    """Minimal evaluator arbitration helper for structured agents."""

    def __init__(self, mode: str = "fixed", learning_rate: float = 0.1) -> None:
        self.mode = mode
        self.learning_rate = float(learning_rate)
        if mode == "adaptive":
            self._weights = np.array([0.4, 0.3, 0.1, 0.2], dtype=float)
        elif mode == "bandit":
            self._weights = np.array([0.5, 0.2, 0.1, 0.2], dtype=float)
        else:
            self._weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        self._update_count = 0
        self._last_signal = 0.0
        self._signal_source = "none"

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        clipped = np.clip(weights, 1e-6, None)
        return clipped / float(np.sum(clipped))

    def _signal_target(self, signal: float, predictive: float, coherence: float, cost: float, horizon: float) -> np.ndarray:
        if signal >= 0.0:
            target = np.array([
                max(predictive, 0.0),
                max(coherence, 0.0),
                max(1.0 / (1.0 + max(cost, 0.0)), 0.0),
                max(horizon, 0.0),
            ], dtype=float)
        else:
            target = np.array([
                max(1.0 - predictive, 0.0),
                max(1.0 - coherence, 0.0),
                max(max(cost, 0.0), 0.0),
                max(1.0 - horizon, 0.0),
            ], dtype=float)
        if float(target.sum()) <= 0.0:
            target = np.full(4, 0.25, dtype=float)
        return self._normalize(target)

    def aggregate(self, predictive: float, coherence: float, cost: float, horizon: float) -> float:
        if self.mode == "adaptive":
            weights = self._weights
            return float(weights[0] * predictive + weights[1] * coherence + weights[2] * cost + weights[3] * horizon)
        if self.mode == "bandit":
            weights = self._weights
            return float(weights[0] * predictive + weights[1] * coherence + weights[2] * horizon + weights[3] * cost)
        return float(predictive + coherence + cost + horizon)

    def update(self, signal: float, predictive: float, coherence: float, cost: float, horizon: float, signal_source: str = "none") -> MetaEvaluatorState:
        self._last_signal = float(signal)
        self._signal_source = signal_source
        if self.mode == "fixed":
            return self.state()

        rate = self.learning_rate * (1.25 if self.mode == "bandit" else 1.0)
        rate = float(np.clip(rate, 0.0, 1.0))
        target = self._signal_target(signal, predictive, coherence, cost, horizon)
        self._weights = self._normalize((1.0 - rate) * self._weights + rate * target)
        self._update_count += 1
        return self.state()

    def state(self) -> MetaEvaluatorState:
        return MetaEvaluatorState(
            mode=self.mode,
            weights=tuple(float(v) for v in self._weights),
            last_signal=float(self._last_signal),
            update_count=self._update_count,
            signal_source=self._signal_source,
        )


class PatternLifecycleTracker:
    """Minimal HPM lifecycle tracker for structured agents."""

    def __init__(
        self,
        consolidation_window: int = 3,
        stable_weight_threshold: float = 0.25,
        retire_weight_threshold: float = 0.05,
        absence_window: int = 3,
        decay_rate: float = 0.1,
    ) -> None:
        self.consolidation_window = consolidation_window
        self.stable_weight_threshold = stable_weight_threshold
        self.retire_weight_threshold = retire_weight_threshold
        self.absence_window = absence_window
        self.decay_rate = decay_rate
        self._identities: dict[str, PatternIdentity] = {}
        self._states: dict[str, PatternState] = {}

    @property
    def identities(self) -> dict[str, PatternIdentity]:
        return self._identities

    @property
    def states(self) -> dict[str, PatternState]:
        return self._states

    def observe(self, pattern, weight: float, step: int) -> None:
        identity = self._identities.get(pattern.id)
        if identity is None:
            parents: tuple[str, ...] = ()
            source_ids = getattr(pattern, "parent_ids", None)
            lineage_kind = "direct"
            if source_ids:
                parents = tuple(str(parent_id) for parent_id in source_ids)
                lineage_kind = "recombined"
            else:
                source_id = getattr(pattern, "source_id", None)
                if source_id:
                    parents = (str(source_id),)
                    lineage_kind = "promoted"
            identity = PatternIdentity(
                id=str(pattern.id),
                parent_ids=parents,
                layer_origin=int(getattr(pattern, "level", 1)),
                created_at=step,
                last_seen_at=step,
                source_step=step,
                lineage_kind=lineage_kind,
            )
            self._identities[pattern.id] = identity

        state = self._states.get(pattern.id)
        if state is None:
            state = PatternState(
                identity_id=pattern.id,
                decay_rate=self.decay_rate,
            )
            self._states[pattern.id] = state

        was_absent = state.absence_count > 0 or state.lifecycle_state in {"decaying", "retired"}
        if state.lifecycle_state == "retired":
            state.lifecycle_state = "emergent"
        if was_absent:
            state.reactivation_count += 1

        state.reinforcement_count += 1
        clipped_weight = float(np.clip(weight, 0.0, 1.0))
        state.stability = float(
            (1.0 - state.decay_rate) * state.stability + state.decay_rate * clipped_weight
        )
        state.decay_rate = self.decay_rate
        identity = self._identities[pattern.id]
        self._identities[pattern.id] = PatternIdentity(
            id=identity.id,
            parent_ids=identity.parent_ids,
            layer_origin=identity.layer_origin,
            created_at=identity.created_at,
            last_seen_at=step,
            source_step=identity.source_step,
            lineage_kind=identity.lineage_kind,
        )

        if clipped_weight >= self.stable_weight_threshold and state.reinforcement_count >= self.consolidation_window:
            state.lifecycle_state = "stable"
        elif clipped_weight <= self.retire_weight_threshold:
            state.lifecycle_state = "decaying"
        else:
            state.lifecycle_state = "stable" if state.reinforcement_count >= self.consolidation_window else "emergent"

    def finalize(self, active_pattern_ids: set[str], step: int) -> None:
        for pattern_id, state in self._states.items():
            if pattern_id in active_pattern_ids:
                state.absence_count = 0
                continue
            identity = self._identities.get(pattern_id)
            if identity is None:
                continue
            state.absence_count += 1
            if step - identity.last_seen_at >= self.absence_window:
                state.lifecycle_state = "retired"
            else:
                state.lifecycle_state = "decaying"
            state.stability = float((1.0 - state.decay_rate) * state.stability)

    def summary(self) -> LifecycleSummary:
        summary = LifecycleSummary()
        for state in self._states.values():
            if state.lifecycle_state == "stable":
                summary.stable += 1
            elif state.lifecycle_state == "decaying":
                summary.decaying += 1
            elif state.lifecycle_state == "retired":
                summary.retired += 1
            else:
                summary.emergent += 1
        return summary

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            pattern_id: {
                "identity": self._identities[pattern_id].to_dict(),
                "state": state.to_dict(),
            }
            for pattern_id, state in self._states.items()
            if pattern_id in self._identities
        }
