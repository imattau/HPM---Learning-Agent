from hpm_ai_v5.evaluation.objective import V5ObjectiveEvaluator
from hpm_ai_v5.planning import OpenAdapterDiscoveryBenchmark


def test_open_adapter_discovery_benchmark_passes() -> None:
    benchmark = OpenAdapterDiscoveryBenchmark().run()

    assert benchmark.result == "success"
    assert benchmark.reason == "success"
    assert benchmark.learned_profiles >= 2

    task_results = {result.name: result for result in benchmark.task_results}

    assert task_results["NumericTest"].selected_pipeline == "numeric_combo"
    assert task_results["NumericTest"].matched
    assert task_results["NumericTest"].profile_reused
    assert task_results["NumericTest"].query_accuracy >= 0.8

    assert task_results["GridTest"].selected_pipeline == "grid_combo"
    assert task_results["GridTest"].matched
    assert task_results["GridTest"].profile_reused
    assert task_results["GridTest"].query_accuracy >= 0.8

    assert task_results["GraphHeldOut"].deferred
    assert task_results["GraphHeldOut"].matched


def test_objective_includes_open_adapter_discovery() -> None:
    report = V5ObjectiveEvaluator().evaluate()
    benchmark_scores = {benchmark.name: benchmark for benchmark in report.benchmarks}

    assert "open_adapter_discovery" in benchmark_scores
    assert benchmark_scores["open_adapter_discovery"].passed
