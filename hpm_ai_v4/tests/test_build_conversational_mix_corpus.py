from hpm_ai_v4.simulations.build_conversational_mix_corpus import (
    DialogueSourceSpec,
    build_conversational_mix_corpus,
)


def test_build_conversational_mix_corpus_uses_multiple_sources(tmp_path, monkeypatch):
    rows = [
        {"dialog": ["Hello there.", "Hi.", "How are you?", "Fine."]},
        {"dialogue": [{"role": "user", "content": "What is this?"}, {"role": "assistant", "content": "A test."}]},
    ]
    calls = []

    def fake_load(repo_id, parquet_file):
        calls.append((repo_id, parquet_file))
        return rows

    monkeypatch.setattr("hpm_ai_v4.simulations.build_conversational_mix_corpus._load_parquet_dataset", fake_load)

    result = build_conversational_mix_corpus(
        output_path=str(tmp_path / "mixed.txt"),
        sources=[
            DialogueSourceSpec(name="one", repo_id="repo/one", parquet_file="one.parquet", row_limit=1, max_turns=4),
            DialogueSourceSpec(name="two", repo_id="repo/two", parquet_file="two.parquet", row_limit=1, max_turns=4),
        ],
        max_dialogs_total=2,
        max_turns=4,
    )

    assert result.dialogs_written == 2
    assert result.lines_written >= 4
    assert result.source_counts == {"one": 1, "two": 1}
    assert len(calls) == 2
    assert (tmp_path / "mixed.txt").exists()
