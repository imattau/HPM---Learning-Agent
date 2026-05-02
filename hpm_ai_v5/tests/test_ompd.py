from hpm_ai_v5.planning import OnlineMetaPatternDiscoveryBenchmark


def test_ompd_discovers_and_reuses_meta_pattern() -> None:
    result = OnlineMetaPatternDiscoveryBenchmark().run()

    assert result.result == "success"
    assert result.meta_pattern_count >= 1
    assert result.test_ratio >= 0.8
    assert result.training_mean_ratio >= 0.8
    assert result.task_results[-1].plan == ["collect_rune", "collect_altar", "collect_gate"]
