"""Promotion ledger unit tests."""

from __future__ import annotations

from pathlib import Path

from hpm.decomposition import CompositePattern
from hpm.promotion import PromotionLedger, PromotionRule


def _sample_pattern() -> CompositePattern:
    return CompositePattern(
        pattern_id="prom_test",
        part_ids=("leg", "body"),
        relation_ids=("leg->body",),
        assembly_rule="test_assembly",
        score=0.42,
        stability=0.64,
        parent_pattern_ids=(),
        trace_id="trace_promo",
    )


def test_promotion_ledger_promotes_and_persists(tmp_path: Path) -> None:
    rule = PromotionRule(min_occurrences=2, min_support=0.5, max_ambiguity=0.9, min_delta_lift=0.0)
    ledger = PromotionLedger(tmp_path / "promo.sqlite", rule=rule)
    pattern = _sample_pattern()

    ledger.record_occurrence(
        task_id="task_alpha",
        trace_id="trace_a",
        pattern=pattern,
        baseline_score=1.0,
        final_score=0.9,
        coverage=0.6,
        ambiguity_rate=0.2,
        selected=True,
        promotion_bonus=0.0,
    )
    ledger.record_occurrence(
        task_id="task_beta",
        trace_id="trace_b",
        pattern=pattern,
        baseline_score=1.0,
        final_score=0.85,
        coverage=0.7,
        ambiguity_rate=0.15,
        selected=True,
    )

    snapshot = ledger.snapshot()
    assert snapshot.promoted_count == 1
    assert snapshot.total_occurrences == 2
    assert snapshot.reuse_rate == 0.0

    bonus, reuse_ids, signature = ledger.score_bonus(pattern)
    assert bonus > 0.0
    assert signature
    assert reuse_ids

    reloaded = PromotionLedger(tmp_path / "promo.sqlite")
    assert len(reloaded.promoted_patterns()) == 1
    assert reloaded.has_promoted(signature)
    assert reloaded.snapshot().promoted_count == 1
