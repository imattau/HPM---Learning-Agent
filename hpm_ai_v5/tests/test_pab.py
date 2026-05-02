from hpm_ai_v5.planning import PolygraphAgreementBenchmark


def test_pab_prefers_agreement_over_noisy_view() -> None:
    result = PolygraphAgreementBenchmark().run()

    assert result.result == "success"
    assert result.agreement_accuracy >= 0.95
    assert result.agreement_accuracy >= result.noisy_accuracy + 0.15
    assert result.consensus_rate >= 0.7
    assert result.steps
    assert any(step.selected_view in {"exact", "trend"} for step in result.steps)
