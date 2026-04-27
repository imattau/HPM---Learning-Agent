from hpm_ai_v4.simulations import build_dailydialog_corpus as dd
from hpm_ai_v4.simulations.dailydialog_polygraph_benchmark import run_dailydialog_polygraph_benchmark


def test_dailydialog_polygraph_benchmark_prefers_structured_memory(tmp_path, monkeypatch):
    fake_dataset = [
        {"dialog": ["Hello there.", "Hi.", "How are you doing?"], "act": [0, 1, 0], "emotion": [0, 0, 0]},
        {"dialog": ["Can you help me?", "Sure.", "What do you need?"], "act": [0, 1, 0], "emotion": [0, 0, 0]},
        {"dialog": ["Good morning.", "Morning.", "I need clarification."], "act": [0, 1, 0], "emotion": [0, 0, 0]},
        {"dialog": ["What is the plan?", "We can start soon.", "Any questions?"], "act": [0, 1, 0], "emotion": [0, 0, 0]},
        {"dialog": ["Hello again.", "Hi again.", "What can I ask?"], "act": [0, 1, 0], "emotion": [0, 0, 0]},
        {"dialog": ["Need more info.", "I can provide it.", "Tell me more."], "act": [0, 1, 0], "emotion": [0, 0, 0]},
    ]
    monkeypatch.setattr(dd, "_load_dataset", lambda name, split: fake_dataset)

    result = run_dailydialog_polygraph_benchmark(
        output_dir=str(tmp_path),
        train_limit=4,
        eval_limit=2,
        max_turns=3,
    )

    assert result.structured.field_match_rate >= result.baseline.field_match_rate
    assert result.structured.exact_match_rate >= result.baseline.exact_match_rate
    assert result.structured.dominant_intent_match_rate >= result.baseline.dominant_intent_match_rate
    assert (tmp_path / "structured_polygraph_bundle.l1.pkl").exists()
    assert (tmp_path / "baseline_polygraph_bundle.l1.pkl").exists()
    assert (tmp_path / "polygraph_comparison.json").exists()
