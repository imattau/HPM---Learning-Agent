from hpm_ai_v5.planning.melb import MultiEpisodeLifecycleBenchmark


def test_multi_episode_lifecycle() -> None:
    result = MultiEpisodeLifecycleBenchmark().run()
    assert result.result == "success", (
        f"MELB failed: {result.reason}. "
        f"transfer_gain={result.transfer_accuracy_gain:.2f} "
        f"isolation={result.context_isolation_verified} "
        f"ep3_seeded={result.episodes[2].patterns_seeded}"
    )
