"""Smoke tests for the ARC-AGI-2 completion comparison benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks import arc_agi_2_compare as arc2


def _write_task(path: Path, value: int) -> None:
    task = {
        "train": [
            {
                "input": [[0, value], [value, 0]],
                "output": [[value, 0], [0, value]],
            }
        ],
        "test": [
            {
                "input": [[0, value], [value, 0]],
                "output": [[value, 0], [0, value]],
            }
        ],
    }
    path.write_text(json.dumps(task), encoding="utf-8")


def test_arc_agi_2_loader_and_run_smoke(tmp_path: Path):
    data_dir = tmp_path / "data" / "evaluation"
    data_dir.mkdir(parents=True)
    for idx in range(5):
        _write_task(data_dir / f"task_{idx}.json", idx + 1)

    tasks = arc2.load_tasks(root=tmp_path, split="evaluation")
    assert len(tasks) == 5
    assert all(arc2.task_fits(task) for task in tasks)

    result = arc2.run(split="evaluation", root=tmp_path, max_tasks=5, seed=7)
    assert result["tasks_run"] == 5
    assert result["excluded"] == 0
    assert result["stack_levels"] == 5
    assert set(result) >= {"baseline", "completion", "delta", "tasks_run", "excluded", "split", "stack_levels"}
    expected_keys = {
        "accuracy",
        "correct",
        "total",
        "predictions",
        "lineage_integrity",
        "trace_completeness",
        "evaluator_drift",
        "decomposition_coverage",
        "decomposition_ambiguity_rate",
        "decomposition_alignment",
        "assembly_quality",
        "promotion_total_occurrences",
        "promotion_promoted_count",
        "promotion_reused_count",
        "promotion_rate",
        "reuse_rate",
        "promotion_retire_count",
        "promotion_bonus_mean",
        "promotion_insights",
    }
    assert set(result["baseline"]) >= expected_keys
    assert set(result["completion"]) >= expected_keys
    assert isinstance(result["baseline"]["predictions"], tuple)
    assert isinstance(result["completion"]["predictions"], tuple)
    assert isinstance(result["baseline"]["promotion_insights"], list)
    assert isinstance(result["completion"]["promotion_insights"], list)


def test_arc_agi_2_ablation_sweep_smoke(tmp_path: Path):
    data_dir = tmp_path / "data" / "evaluation"
    data_dir.mkdir(parents=True)
    for idx in range(5):
        _write_task(data_dir / f"task_{idx}.json", idx + 1)

    result = arc2.run_ablation_sweep(split="evaluation", root=tmp_path, max_tasks=5, seed=7)
    assert result["tasks_run"] == 5
    assert result["excluded"] == 0
    assert result["stack_levels"] == 5
    assert set(result["conditions"]) == {
        "baseline",
        "completion",
        "completion_no_identity",
        "completion_no_constraints",
        "completion_no_meta_eval",
    }
    assert set(result["delta"]) == {
        "completion",
        "completion_no_identity",
        "completion_no_constraints",
        "completion_no_meta_eval",
    }
    expected_delta_keys = {
        "accuracy",
        "lineage_integrity",
        "trace_completeness",
        "evaluator_drift",
        "decomposition_coverage",
        "decomposition_ambiguity_rate",
        "decomposition_alignment",
        "assembly_quality",
        "promotion_rate",
        "reuse_rate",
        "promotion_promoted_count",
        "promotion_bonus_mean",
    }
    for delta in result["delta"].values():
        assert set(delta) >= expected_delta_keys



def test_arc_agi_2_causal_helpers():
    simple = [[1, 0], [0, 1]]
    complex_grid = [[1, 2], [3, 4]]
    simple_complexity = arc2._grid_complexity(simple)
    complex_complexity = arc2._grid_complexity(complex_grid)
    assert complex_complexity > simple_complexity

    from hpm.completion_types import FieldConstraint

    prefer_simple = [FieldConstraint("prefer_simple", "*", 0.8, "test", 1)]
    simple_adjustment = arc2._field_constraint_adjustment(simple, prefer_simple)
    complex_adjustment = arc2._field_constraint_adjustment(complex_grid, prefer_simple)
    assert simple_adjustment < complex_adjustment


def test_arc_agi_2_identity_trust_helper():
    low = [{"a": {"state": {"lifecycle_state": "emergent"}, "identity": {"last_seen_at": 0}}}]
    high = [{"a": {"state": {"lifecycle_state": "stable"}, "identity": {"last_seen_at": 25}}}]
    assert arc2._identity_trust(high) > arc2._identity_trust(low)



def test_arc_agi_2_rank_margin_reranker_prefers_consensus():
    candidate_rows = [
        {
            "base": 1.02,
            "l1": 0.05,
            "l2": 0.08,
            "l3": 0.06,
            "core": 0.04,
            "identity_trust": 0.95,
            "meta_trust": 0.90,
            "structure_trust": 0.85,
            "candidate_complexity": 0.10,
        },
        {
            "base": 1.00,
            "l1": 0.75,
            "l2": 0.70,
            "l3": 0.68,
            "core": 0.72,
            "identity_trust": 0.10,
            "meta_trust": 0.10,
            "structure_trust": 0.10,
            "candidate_complexity": 0.90,
        },
    ]
    reweighted = arc2._apply_rank_margin_reweight(candidate_rows, True)
    assert reweighted[0] < reweighted[1]
    assert arc2._apply_rank_margin_reweight(candidate_rows, False) == [1.02, 1.0]
