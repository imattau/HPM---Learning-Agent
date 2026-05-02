from hpm_ai_v5.planning import RotatingSequenceGeneralizationBenchmark


def test_rsg_generalizes_rotating_period_three() -> None:
    benchmark = RotatingSequenceGeneralizationBenchmark()

    result = benchmark.evaluate((1.0, 2.0, 3.0), phases=(0, 1, 2), horizon=3)

    assert result.result == "success"
    assert all(phase_result.sequence_length == 3 for phase_result in result.phase_results)
    assert all(phase_result.accuracy >= 0.9 for phase_result in result.phase_results)


def test_rsg_generalizes_rotating_period_four() -> None:
    benchmark = RotatingSequenceGeneralizationBenchmark()

    result = benchmark.evaluate((1.0, 2.0, 3.0, 4.0), phases=(0, 1, 2, 3), horizon=4)

    assert result.result == "success"
    assert all(phase_result.sequence_length == 4 for phase_result in result.phase_results)
    assert all(phase_result.accuracy >= 0.9 for phase_result in result.phase_results)
    assert all(phase_result.reasoning_trace["selected_sequence"] for phase_result in result.phase_results)
