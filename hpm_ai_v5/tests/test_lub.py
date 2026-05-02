from hpm_ai_v5.planning import LearnedUtilityBenchmark


def test_lub_learns_contextual_utility_preferences() -> None:
    result = LearnedUtilityBenchmark().run()

    assert result.result == "success"
    assert result.preference_accuracy >= 0.8
    assert result.utility_separation >= 0.6
    assert result.episode_results
    assert any(episode.selected_pattern == "Q_risky" for episode in result.episode_results if episode.context == 0)
    assert any(episode.selected_pattern == "P_safe" for episode in result.episode_results if episode.context == 1)
