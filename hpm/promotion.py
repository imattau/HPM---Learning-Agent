
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from hpm.decomposition import CompositePattern, TaskDecompositionResult


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True)


def _now() -> float:
    return float(time.time())


def _signature_bucket(value: float, scale: float = 0.25) -> int:
    return int(np.clip(round(float(value) / max(scale, 1e-6)), 0, 12))


def composite_signature(pattern: CompositePattern) -> str:
    """Stable recurrence signature for a composite pattern."""
    if pattern.assembly_rule == "task_summary":
        return ""
    part_bucket = min(len(pattern.part_ids), 8)
    relation_bucket = min(len(pattern.relation_ids), 12)
    score_bucket = _signature_bucket(pattern.score, 0.2)
    stability_bucket = _signature_bucket(pattern.stability, 0.2)
    return f"{pattern.assembly_rule}|p{part_bucket}|r{relation_bucket}|s{score_bucket}|t{stability_bucket}"


def best_promotable_pattern(result: TaskDecompositionResult) -> CompositePattern | None:
    candidates = [pattern for pattern in result.candidate_patterns if pattern.assembly_rule != "task_summary"]
    if not candidates:
        candidates = list(result.candidate_patterns)
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.score, item.stability, item.pattern_id))


@dataclass(frozen=True)
class CompositeOccurrence:
    occurrence_id: str
    task_id: str
    trace_id: str
    composite_signature: str
    source_part_ids: tuple[str, ...]
    source_relation_ids: tuple[str, ...]
    candidate_pattern_id: str
    baseline_score: float
    final_score: float
    coverage: float
    ambiguity_rate: float
    selected: bool
    promotion_bonus: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotedPattern:
    pattern_id: str
    signature: str
    label: str
    origin_composite_ids: tuple[str, ...]
    parent_part_ids: tuple[str, ...]
    promotion_count: int
    support: float
    stability: float
    retention_state: str
    last_seen_at: float
    trace_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionRule:
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    min_occurrences: int = 2
    min_support: float = 0.55
    max_ambiguity: float = 0.65
    min_delta_lift: float = 0.02
    retention_window: int = 5
    selected_weight: float = 1.0
    unselected_weight: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionTrace:
    trace_id: str
    task_id: str
    composite_id: str
    promotion_decision: str
    decision_reason: str
    support_breakdown: dict[str, float]
    lineage: tuple[str, ...]
    candidate_reuse: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionLedgerSnapshot:
    total_occurrences: int
    promoted_count: int
    reused_count: int
    promotion_rate: float
    reuse_rate: float
    retire_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PromotionLedger:
    """Store-backed recurrence and promotion ledger for composite patterns."""

    def __init__(self, storage_path: str | Path | None = None, rule: PromotionRule | None = None) -> None:
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self.rule = rule or PromotionRule()
        self._occurrences: dict[str, list[CompositeOccurrence]] = {}
        self._promoted: dict[str, PromotedPattern] = {}
        self._promotion_traces: list[PromotionTrace] = []
        self._reused_hits: int = 0
        self._retired: set[str] = set()
        if self.storage_path is not None:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._load_state()

    def _connect(self) -> sqlite3.Connection:
        if self.storage_path is None:
            raise RuntimeError("PromotionLedger has no storage_path")
        return sqlite3.connect(str(self.storage_path))

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS composite_occurrences (
                    occurrence_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    composite_signature TEXT NOT NULL,
                    source_part_ids TEXT NOT NULL,
                    source_relation_ids TEXT NOT NULL,
                    candidate_pattern_id TEXT NOT NULL,
                    baseline_score REAL NOT NULL,
                    final_score REAL NOT NULL,
                    coverage REAL NOT NULL,
                    ambiguity_rate REAL NOT NULL,
                    selected INTEGER NOT NULL,
                    promotion_bonus REAL NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS promoted_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    signature TEXT NOT NULL,
                    label TEXT NOT NULL,
                    origin_composite_ids TEXT NOT NULL,
                    parent_part_ids TEXT NOT NULL,
                    promotion_count INTEGER NOT NULL,
                    support REAL NOT NULL,
                    stability REAL NOT NULL,
                    retention_state TEXT NOT NULL,
                    last_seen_at REAL NOT NULL,
                    trace_id TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS promotion_traces (
                    trace_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    composite_id TEXT NOT NULL,
                    promotion_decision TEXT NOT NULL,
                    decision_reason TEXT NOT NULL,
                    support_breakdown TEXT NOT NULL,
                    lineage TEXT NOT NULL,
                    candidate_reuse TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _load_state(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT occurrence_id, task_id, trace_id, composite_signature, source_part_ids, source_relation_ids, candidate_pattern_id, baseline_score, final_score, coverage, ambiguity_rate, selected, promotion_bonus FROM composite_occurrences"
            ).fetchall()
            for row in rows:
                occurrence = CompositeOccurrence(
                    occurrence_id=row[0],
                    task_id=row[1],
                    trace_id=row[2],
                    composite_signature=row[3],
                    source_part_ids=tuple(json.loads(row[4])),
                    source_relation_ids=tuple(json.loads(row[5])),
                    candidate_pattern_id=row[6],
                    baseline_score=float(row[7]),
                    final_score=float(row[8]),
                    coverage=float(row[9]),
                    ambiguity_rate=float(row[10]),
                    selected=bool(row[11]),
                    promotion_bonus=float(row[12]),
                )
                self._occurrences.setdefault(occurrence.composite_signature, []).append(occurrence)

            rows = conn.execute(
                "SELECT pattern_id, signature, label, origin_composite_ids, parent_part_ids, promotion_count, support, stability, retention_state, last_seen_at, trace_id FROM promoted_patterns"
            ).fetchall()
            for row in rows:
                promoted = PromotedPattern(
                    pattern_id=row[0],
                    signature=row[1],
                    label=row[2],
                    origin_composite_ids=tuple(json.loads(row[3])),
                    parent_part_ids=tuple(json.loads(row[4])),
                    promotion_count=int(row[5]),
                    support=float(row[6]),
                    stability=float(row[7]),
                    retention_state=row[8],
                    last_seen_at=float(row[9]),
                    trace_id=row[10],
                )
                self._promoted[promoted.signature] = promoted
            self._reused_hits = int(sum(1 for occs in self._occurrences.values() for occ in occs if occ.promotion_bonus > 0.0))
        finally:
            conn.close()

    def signature_for(self, pattern: CompositePattern) -> str:
        return composite_signature(pattern)

    def promoted_patterns(self) -> list[PromotedPattern]:
        return list(self._promoted.values())

    def has_promoted(self, signature: str) -> bool:
        return signature in self._promoted

    def score_bonus(self, pattern: CompositePattern) -> tuple[float, tuple[str, ...], str]:
        signature = self.signature_for(pattern)
        promoted = self._promoted.get(signature)
        if promoted is None:
            return 0.0, tuple(), signature
        bonus = min(
            0.24,
            0.04 * promoted.promotion_count
            + 0.08 * promoted.support
            + 0.08 * promoted.stability,
        )
        return float(bonus), (promoted.pattern_id,), signature

    def record_occurrence(
        self,
        *,
        task_id: str,
        trace_id: str,
        pattern: CompositePattern,
        baseline_score: float,
        final_score: float,
        coverage: float,
        ambiguity_rate: float,
        selected: bool,
        source_part_ids: tuple[str, ...] = (),
        source_relation_ids: tuple[str, ...] = (),
        promotion_bonus: float = 0.0,
    ) -> CompositeOccurrence:
        signature = self.signature_for(pattern)
        occurrence = CompositeOccurrence(
            occurrence_id=f"occ_{uuid.uuid4().hex}",
            task_id=task_id,
            trace_id=trace_id,
            composite_signature=signature,
            source_part_ids=source_part_ids or pattern.part_ids,
            source_relation_ids=source_relation_ids or pattern.relation_ids,
            candidate_pattern_id=pattern.pattern_id,
            baseline_score=float(baseline_score),
            final_score=float(final_score),
            coverage=float(coverage),
            ambiguity_rate=float(ambiguity_rate),
            selected=bool(selected),
            promotion_bonus=float(promotion_bonus),
        )
        self._occurrences.setdefault(signature, []).append(occurrence)
        if self.storage_path is not None:
            self._write_occurrence(occurrence)
        self._maybe_promote(signature, task_id=task_id, trace_id=trace_id)
        if promotion_bonus > 0.0:
            self._reused_hits += 1
        return occurrence

    def _write_occurrence(self, occurrence: CompositeOccurrence) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO composite_occurrences VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    occurrence.occurrence_id,
                    occurrence.task_id,
                    occurrence.trace_id,
                    occurrence.composite_signature,
                    _json(list(occurrence.source_part_ids)),
                    _json(list(occurrence.source_relation_ids)),
                    occurrence.candidate_pattern_id,
                    occurrence.baseline_score,
                    occurrence.final_score,
                    occurrence.coverage,
                    occurrence.ambiguity_rate,
                    1 if occurrence.selected else 0,
                    occurrence.promotion_bonus,
                    _now(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _write_promoted(self, promoted: PromotedPattern) -> None:
        if self.storage_path is None:
            return
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO promoted_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    promoted.pattern_id,
                    promoted.signature,
                    promoted.label,
                    _json(list(promoted.origin_composite_ids)),
                    _json(list(promoted.parent_part_ids)),
                    promoted.promotion_count,
                    promoted.support,
                    promoted.stability,
                    promoted.retention_state,
                    promoted.last_seen_at,
                    promoted.trace_id,
                    _now(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _write_trace(self, trace: PromotionTrace) -> None:
        self._promotion_traces.append(trace)
        if self.storage_path is None:
            return
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO promotion_traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.trace_id,
                    trace.task_id,
                    trace.composite_id,
                    trace.promotion_decision,
                    trace.decision_reason,
                    _json(trace.support_breakdown),
                    _json(list(trace.lineage)),
                    _json(list(trace.candidate_reuse)),
                    _now(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _maybe_promote(self, signature: str, *, task_id: str, trace_id: str) -> PromotedPattern | None:
        if not signature:
            return None
        occs = self._occurrences.get(signature, [])
        if not occs:
            return None
        selected = [occ for occ in occs if occ.selected]
        support = len(selected) / len(occs)
        mean_ambiguity = float(np.mean([occ.ambiguity_rate for occ in occs])) if occs else 1.0
        mean_lift = float(np.mean([max(0.0, occ.baseline_score - occ.final_score) for occ in selected])) if selected else 0.0
        mean_score = float(np.mean([occ.final_score for occ in selected])) if selected else float(np.mean([occ.final_score for occ in occs]))
        if len(selected) < self.rule.min_occurrences:
            self._write_trace(
                PromotionTrace(
                    trace_id=f"trace_{uuid.uuid4().hex}",
                    task_id=task_id,
                    composite_id=signature,
                    promotion_decision="defer",
                    decision_reason="insufficient_occurrences",
                    support_breakdown={
                        "occurrences": float(len(occs)),
                        "selected_occurrences": float(len(selected)),
                        "support": support,
                        "mean_ambiguity": mean_ambiguity,
                        "mean_lift": mean_lift,
                    },
                    lineage=tuple(occ.candidate_pattern_id for occ in selected),
                    candidate_reuse=tuple(occ.candidate_pattern_id for occ in occs if occ.promotion_bonus > 0.0),
                )
            )
            return None
        if support < self.rule.min_support:
            decision = "reject"
            reason = "low_support"
        elif mean_ambiguity > self.rule.max_ambiguity:
            decision = "reject"
            reason = "high_ambiguity"
        elif mean_lift < self.rule.min_delta_lift:
            decision = "reject"
            reason = "insufficient_lift"
        else:
            decision = "promote"
            reason = "threshold_met"

        if decision != "promote":
            self._write_trace(
                PromotionTrace(
                    trace_id=f"trace_{uuid.uuid4().hex}",
                    task_id=task_id,
                    composite_id=signature,
                    promotion_decision=decision,
                    decision_reason=reason,
                    support_breakdown={
                        "occurrences": float(len(occs)),
                        "selected_occurrences": float(len(selected)),
                        "support": support,
                        "mean_ambiguity": mean_ambiguity,
                        "mean_lift": mean_lift,
                    },
                    lineage=tuple(occ.candidate_pattern_id for occ in selected),
                    candidate_reuse=tuple(occ.candidate_pattern_id for occ in occs if occ.promotion_bonus > 0.0),
                )
            )
            return None

        promoted = self._promoted.get(signature)
        if promoted is None:
            promoted = PromotedPattern(
                pattern_id=f"prom_{uuid.uuid4().hex}",
                signature=signature,
                label=signature.split("|", 1)[0],
                origin_composite_ids=tuple(occ.candidate_pattern_id for occ in selected),
                parent_part_ids=tuple(dict.fromkeys(pid for occ in selected for pid in occ.source_part_ids)),
                promotion_count=len(selected),
                support=float(np.clip(support, 0.0, 1.0)),
                stability=float(np.clip(1.0 - mean_ambiguity, 0.0, 1.0)),
                retention_state="promoted",
                last_seen_at=_now(),
                trace_id=trace_id,
            )
        else:
            promoted = PromotedPattern(
                pattern_id=promoted.pattern_id,
                signature=promoted.signature,
                label=promoted.label,
                origin_composite_ids=tuple(dict.fromkeys((*promoted.origin_composite_ids, *[occ.candidate_pattern_id for occ in selected]))),
                parent_part_ids=tuple(dict.fromkeys((*promoted.parent_part_ids, *[pid for occ in selected for pid in occ.source_part_ids]))),
                promotion_count=promoted.promotion_count + len(selected),
                support=float(np.clip(max(promoted.support, support), 0.0, 1.0)),
                stability=float(np.clip(max(promoted.stability, 1.0 - mean_ambiguity), 0.0, 1.0)),
                retention_state="promoted",
                last_seen_at=_now(),
                trace_id=trace_id,
            )
        self._promoted[signature] = promoted
        self._write_promoted(promoted)
        self._write_trace(
            PromotionTrace(
                trace_id=f"trace_{uuid.uuid4().hex}",
                task_id=task_id,
                composite_id=signature,
                promotion_decision="promote",
                decision_reason=reason,
                support_breakdown={
                    "occurrences": float(len(occs)),
                    "selected_occurrences": float(len(selected)),
                    "support": support,
                    "mean_ambiguity": mean_ambiguity,
                    "mean_lift": mean_lift,
                    "mean_score": mean_score,
                },
                lineage=tuple(occ.candidate_pattern_id for occ in selected),
                candidate_reuse=tuple(occ.candidate_pattern_id for occ in occs if occ.promotion_bonus > 0.0),
            )
        )
        return promoted

    def promote(self, *, task_id: str, trace_id: str, pattern: CompositePattern, selected: bool, baseline_score: float, final_score: float, coverage: float, ambiguity_rate: float, source_part_ids: tuple[str, ...] = (), source_relation_ids: tuple[str, ...] = (), promotion_bonus: float = 0.0) -> CompositeOccurrence:
        return self.record_occurrence(
            task_id=task_id,
            trace_id=trace_id,
            pattern=pattern,
            baseline_score=baseline_score,
            final_score=final_score,
            coverage=coverage,
            ambiguity_rate=ambiguity_rate,
            selected=selected,
            source_part_ids=source_part_ids,
            source_relation_ids=source_relation_ids,
            promotion_bonus=promotion_bonus,
        )

    def snapshot(self) -> PromotionLedgerSnapshot:
        total_occurrences = sum(len(v) for v in self._occurrences.values())
        promoted_count = len(self._promoted)
        reused_count = self._reused_hits
        promotion_rate = promoted_count / max(total_occurrences, 1)
        reuse_rate = reused_count / max(total_occurrences, 1)
        retire_count = len(self._retired)
        return PromotionLedgerSnapshot(
            total_occurrences=total_occurrences,
            promoted_count=promoted_count,
            reused_count=reused_count,
            promotion_rate=float(promotion_rate),
            reuse_rate=float(reuse_rate),
            retire_count=retire_count,
        )

    def promoted_patterns_for_reuse(self) -> list[PromotedPattern]:
        return [pattern for pattern in self._promoted.values() if pattern.retention_state == "promoted"]


__all__ = [
    "CompositeOccurrence",
    "PromotedPattern",
    "PromotionRule",
    "PromotionTrace",
    "PromotionLedger",
    "PromotionLedgerSnapshot",
    "best_promotable_pattern",
    "composite_signature",
]
