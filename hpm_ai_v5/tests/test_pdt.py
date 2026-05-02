from hpm_ai_v5.planning import PrefixDisambiguationTask


def test_pdt_exposes_prefix_disambiguation_gap() -> None:
    result = PrefixDisambiguationTask().run()

    assert result.result == "failure"
    assert result.critical_accuracy < 0.7
    assert result.accuracy == result.critical_accuracy
    assert result.step_results
    assert any(step.prefix[:2] == ["C", "A"] or step.prefix[:2] == ["A", "A"] for step in result.step_results)

def test_pdt_buffered_adapter_resolves_ambiguity() -> None:
    result = PrefixDisambiguationTask(use_prefix_buffer=True).run()

    assert result.result == "success"
    assert result.critical_accuracy >= 0.7
    assert result.accuracy == result.critical_accuracy
    assert result.step_results
