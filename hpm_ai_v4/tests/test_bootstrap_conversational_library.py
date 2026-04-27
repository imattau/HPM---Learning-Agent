from hpm_ai_v4.simulations.bootstrap_conversational_library import bootstrap_conversational_library


def test_bootstrap_conversational_library_registers_seed(tmp_path, monkeypatch):
    daily = tmp_path / "daily_dialog_chat_corpus.txt"
    daily.write_text("User: hello\nAssistant: hi\n")
    (tmp_path / "chat_seed.txt").write_text("hello there\n")
    (tmp_path / "wiki_sample.txt").write_text("the quick brown fox\n")
    (tmp_path / "peter_rabbit.txt").write_text("once upon a time\n")
    (tmp_path / "history_of_science.txt").write_text("science history\n")

    def fake_build_library(**kwargs):
        (tmp_path / "conversational_chat_library.pkl").write_text("stub")
        return 0

    monkeypatch.setattr("hpm_ai_v4.simulations.bootstrap_conversational_library.build_library", fake_build_library)
    monkeypatch.setattr(
        "hpm_ai_v4.simulations.bootstrap_conversational_library._local_corpus_sources",
        lambda: [
            str(daily),
            str(tmp_path / "chat_seed.txt"),
            str(tmp_path / "wiki_sample.txt"),
            str(tmp_path / "peter_rabbit.txt"),
            str(tmp_path / "history_of_science.txt"),
        ],
    )

    result = bootstrap_conversational_library(output_dir=str(tmp_path))

    assert result.dialogs_written == 6
    assert result.lines_written >= 5
    assert (tmp_path / "conversational_chat_library.pkl").exists()
