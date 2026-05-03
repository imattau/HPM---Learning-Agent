from __future__ import annotations

from hpm_ai_v5.arc.benchmark import ArcFullBenchmark


def test_arc_full_benchmark_smoke() -> None:
    result = ArcFullBenchmark().run(limit=10, timeout_seconds=1.0)

    assert result.task_count == 10
    assert result.failure_count >= 0
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.acceptance_rate <= 1.0
    assert len(result.task_results) == 10
