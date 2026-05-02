from hpm_ai_v5.planning import DelayedConsequenceMazeBenchmark


def test_dcm_benchmark_passes_and_discovers_sequence() -> None:
    result = DelayedConsequenceMazeBenchmark().run()

    assert result.result == "success"
    assert result.training_reward > 0.0
    assert result.evaluation_reward > 0.0
    assert result.phase_results[-1].sequence_length == 3
    assert result.phase_results[-1].predicted_first_action == -1.0
    assert result.phase_results[-1].selected_sequence is not None
    assert result.phase_results[-1].selected_sequence == ["pattern_2", "pattern_1", "pattern_1"]
