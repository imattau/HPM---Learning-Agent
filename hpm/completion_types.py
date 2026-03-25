from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PatternIdentity:
    id: str
    parent_ids: tuple[str, ...] = ()
    layer_origin: int = 1
    created_at: int = 0
    last_seen_at: int = 0
    source_step: int = 0
    lineage_kind: str = "direct"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatternState:
    identity_id: str
    lifecycle_state: str = "emergent"
    stability: float = 0.0
    decay_rate: float = 0.1
    reinforcement_count: int = 0
    absence_count: int = 0
    reactivation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluatorVector:
    predictive: float
    coherence: float
    cost: float
    horizon: float
    aggregate: float
    arbitration_mode: str = "fixed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetaEvaluatorState:
    mode: str = "fixed"
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    last_signal: float = 0.0
    update_count: int = 0
    signal_source: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FieldConstraint:
    constraint_type: str
    scope: str
    strength: float
    source: str
    timestamp: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionTrace:
    trace_id: str
    selected_pattern_ids: tuple[str, ...]
    selected_parent_ids: tuple[str, ...]
    evaluator_vector: EvaluatorVector
    constraint_ids: tuple[str, ...]
    meta_evaluator_state: MetaEvaluatorState
    signal_source: str
    action: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evaluator_vector"] = self.evaluator_vector.to_dict()
        data["meta_evaluator_state"] = self.meta_evaluator_state.to_dict()
        return data


@dataclass
class LifecycleSummary:
    emergent: int = 0
    stable: int = 0
    decaying: int = 0
    retired: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
