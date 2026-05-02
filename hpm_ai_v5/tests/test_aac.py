from hpm_ai_v5.planning import AutomaticAdapterCompositionBenchmark


def test_aac_discovers_and_reuses_adapter_pipelines() -> None:
    result = AutomaticAdapterCompositionBenchmark().run()

    assert result.result == "success"
    assert result.learned_profiles >= 1
    assert result.test_mean_accuracy >= 0.8
    assert any(task.selected_pipeline == "state_fusion" for task in result.task_results if task.family == "fusion")
