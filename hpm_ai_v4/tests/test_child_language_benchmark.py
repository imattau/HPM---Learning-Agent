from hpm_ai_v4.simulations.build_child_language_corpus import build_child_language_corpus
from hpm_ai_v4.simulations.child_language_benchmark import run_child_language_benchmark


def test_run_child_language_benchmark_short(tmp_path):
    corpus = tmp_path / "child_corpus.txt"
    build_child_language_corpus(
        str(corpus),
        held_out_nouns=["book"],
        held_out_verbs=["go"],
    )

    result = run_child_language_benchmark(
        corpus_path=str(corpus),
        repeats=2,
        seed=7,
        warmup_chars=40,
        response_steps=12,
        history_window=2,
        num_workers=1,
        use_dict=False,
        checkpoint_dir=str(tmp_path),
    )

    assert "sentence_aware" in result
    assert "sentence_blind" in result
    assert "delta" in result
    assert "runs" in result["sentence_aware"]
    assert "runs" in result["sentence_blind"]
    assert "metrics" in result["sentence_aware"]
    assert "metrics" in result["sentence_blind"]
    assert "avg_sentence_shape_score" in result["sentence_blind"]["metrics"]
    assert (tmp_path / "child_language_benchmark.json").exists()
