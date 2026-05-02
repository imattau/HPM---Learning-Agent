from hpm_ai_v5.planning import ScoringWeightAdaptationBenchmark


def test_swa_learns_environment_specific_scoring_weights() -> None:
    result = ScoringWeightAdaptationBenchmark().run()

    assert result.result == "success"
    assert result.environment_results
    assert all(environment.learned_ratio >= 0.9 for environment in result.environment_results)
    assert any(environment.learned_ratio > environment.baseline_ratio for environment in result.environment_results)
    assert any(environment.learned_weights["gamma"] >= environment.learned_weights["beta"] for environment in result.environment_results)
