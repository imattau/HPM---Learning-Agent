from hpm_ai_v4.simulations import bootstrap_chat_library as bootstrap


def test_bootstrap_chat_library_calls_builders(tmp_path, monkeypatch):
    calls = {}

    class FakeCorpusResult:
        corpus_path = str(tmp_path / "daily_dialog_chat_corpus.txt")
        dialogs_written = 3
        lines_written = 6

    def fake_build_dailydialog_corpus(**kwargs):
        calls["corpus"] = kwargs
        (tmp_path / "daily_dialog_chat_corpus.txt").write_text("User: hello\nAssistant: hi\n")
        return FakeCorpusResult()

    def fake_build_library(**kwargs):
        calls["library"] = kwargs
        (tmp_path / "daily_dialog_chat_library.pkl").write_text("stub")
        return 0

    monkeypatch.setattr(bootstrap, "build_dailydialog_corpus", fake_build_dailydialog_corpus)
    monkeypatch.setattr(bootstrap, "build_library", fake_build_library)

    result = bootstrap.bootstrap_chat_library(
        output_dir=str(tmp_path),
        registry_path=None,
        limit=2,
        max_turns=4,
        steps=10,
        min_density=0.1,
    )

    assert "corpus" in calls
    assert "library" in calls
    assert result.dialogs_written == 3
    assert result.lines_written == 6
    assert (tmp_path / "daily_dialog_chat_library.pkl").exists()
