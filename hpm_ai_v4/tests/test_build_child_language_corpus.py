from hpm_ai_v4.simulations.build_child_language_corpus import build_child_language_corpus


def test_build_child_language_corpus_excludes_held_out_words(tmp_path):
    out = tmp_path / "child_corpus.txt"
    result = build_child_language_corpus(
        str(out),
        held_out_nouns=["book"],
        held_out_verbs=["go"],
    )

    text = out.read_text()
    assert "book" not in text.lower()
    assert "go" not in text.lower()
    assert result.lines_written > 0
    assert result.held_out_words["nouns"] == ["book"]
    assert result.held_out_words["verbs"] == ["go"]
