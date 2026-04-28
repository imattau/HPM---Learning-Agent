from hpm_ai_v4.simulations.article_sentence_benchmark import run_article_sentence_benchmark


def test_run_article_sentence_benchmark_short(tmp_path):
    corpus = tmp_path / "article.txt"
    corpus.write_text(
        "First sentence here. Second sentence follows. Third sentence closes the paragraph.\n\n"
        "Another paragraph starts now. It has a follow up sentence. And a final sentence."
    )

    result = run_article_sentence_benchmark(
        corpus_path=str(corpus),
        prompt_sentences=1,
        target_sentences=1,
        examples=1,
        warmup_chars=10,
        response_steps=8,
        history_window=2,
        num_workers=1,
        use_dict=False,
        checkpoint_dir=str(tmp_path),
    )

    assert "sentence_aware" in result
    assert "sentence_blind" in result
    assert "delta" in result
    assert "avg_sentence_structure_score" in result["sentence_aware"]
    assert "avg_sentence_structure_score" in result["sentence_blind"]
    assert (tmp_path / "article_sentence_benchmark.json").exists()
