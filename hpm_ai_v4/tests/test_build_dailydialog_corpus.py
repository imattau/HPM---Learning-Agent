from hpm_ai_v4.simulations import build_dailydialog_corpus as dd


def test_build_dailydialog_corpus_writes_dialogue(tmp_path, monkeypatch):
    fake_dataset = [
        {"dialog": ["Hello there.", "Hi.", "How are you?"], "act": [0, 1, 0], "emotion": [0, 0, 0]},
        {"dialog": ["What's new?", "Not much.", "Want to talk about code?"], "act": [0, 1, 0], "emotion": [0, 0, 0]},
    ]
    monkeypatch.setattr(dd, "_load_dataset", lambda name, split: fake_dataset)

    output = tmp_path / "daily_dialog.txt"
    result = dd.build_dailydialog_corpus(
        output_path=str(output),
        split="train",
        dataset_name="daily_dialog",
        limit=10,
        max_turns=4,
    )

    assert output.exists()
    text = output.read_text()
    assert "User:" in text
    assert "Assistant:" in text
    assert result.dialogs_written == 2
    assert result.lines_written >= 4
    assert result.preview[:2] == ["User: Hello there.", "Assistant: Hi."]
