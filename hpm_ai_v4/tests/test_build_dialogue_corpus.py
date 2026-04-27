from hpm_ai_v4.simulations.build_dialogue_corpus import build_dialogue_corpus


def test_build_dialogue_corpus_writes_dialogue(tmp_path):
    output = tmp_path / "dialogue_corpus.txt"
    result = build_dialogue_corpus(
        output_path=str(output),
        topics=["text", "code"],
        turns_per_topic=2,
        warmup_chars=64,
        num_workers=1,
        use_dict=False,
        response_steps=16,
    )

    assert output.exists()
    text = output.read_text()
    assert "User:" in text
    assert "Assistant:" in text
    assert len(text.splitlines()) >= 4
    assert result.lines_written == len(text.splitlines())
    assert result.topics == ["text", "code"]
