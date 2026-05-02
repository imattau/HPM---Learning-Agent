from hpm_ai_v5.planning import TripleSequenceDiscoveryBenchmark


def test_tsd_discovers_triple_sequence_and_uses_macro_execution() -> None:
    result = TripleSequenceDiscoveryBenchmark().run()

    assert result.result == "success"
    assert result.discovered_sequence_length >= 3
    assert result.macro_reward > result.baseline_reward
    assert result.reward_gain > 0.0
    assert result.macro_steps
    assert any(step.sequence_length >= 3 for step in result.macro_steps)
