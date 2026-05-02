from hpm_ai_v5.evaluation import V5ObjectiveEvaluator


def test_objective_evaluation_returns_numeric_report() -> None:
    report = V5ObjectiveEvaluator().evaluate()

    assert 0.0 <= report.overall_score <= 1.0
    assert report.benchmarks
    assert report.summary["benchmark_names"] == [benchmark.name for benchmark in report.benchmarks]
    assert report.summary["passed_benchmarks"]
    assert all("score" in benchmark.metrics for benchmark in report.benchmarks)
    assert all(0.0 <= benchmark.score <= 1.0 for benchmark in report.benchmarks)


def test_objective_evaluation_includes_expected_benchmarks() -> None:
    report = V5ObjectiveEvaluator().evaluate()
    names = {benchmark.name for benchmark in report.benchmarks}

    assert {"core_reasoning", "agent_pipeline", "nested_prerequisite_planning", "ctw_rule_discovery", "delayed_consequence_maze", "prefix_disambiguation_task", "learned_utility_benchmark", "triple_sequence_discovery", "arc_subset"} <= names
