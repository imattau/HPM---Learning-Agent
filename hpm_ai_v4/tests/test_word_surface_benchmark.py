from pathlib import Path

from hpm_ai_v4.simulations.build_large_word_library import build_large_word_library
from hpm_ai_v4.simulations.word_surface_benchmark import run_word_surface_ab_benchmark


def test_run_word_surface_ab_benchmark_reports_metrics(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "The quick brown fox jumps over the lazy dog. "
        "The fox observes the dog. "
        "The dog sleeps near the fire. "
        "The fox watches the dog again. "
        "The dog wakes at dawn. "
        "The fox runs into the field. "
        "The fire goes out slowly. "
    )

    result = run_word_surface_ab_benchmark(
        corpus_path=str(corpus),
        prompts=[
            "Hello.",
            "What can you do?",
        ],
        prompt_sentences=1,
        target_sentences=1,
        examples=2,
        warmup_chars=80,
        response_steps=10,
        history_window=3,
        num_workers=1,
        use_dict=False,
        checkpoint_dir=str(tmp_path),
    )

    assert result["word"]["surface_mode"] == "word"
    assert result["ascii"]["surface_mode"] == "ascii"
    assert "child" in result
    assert "continuation" in result["word"]
    assert "chat" in result["word"]
    assert "metrics" in result["word"]["continuation"]
    assert "metrics" in result["word"]["chat"]
    assert "avg_token_agreement" in result["word"]["continuation"]["metrics"]
    assert "response_diversity" in result["word"]["chat"]["metrics"]
    assert "metrics" in result["child"]["word"]
    assert "metrics" in result["child"]["ascii"]
    assert "held_out_token_agreement" in result["child"]["word"]["metrics"]
    assert result["examples"][0]["prompt_sentences"] >= 3
    assert result["examples"][0]["target_sentences"] >= 2
    assert result["preferred_arm"] in {"word", "ascii"}
    assert (tmp_path / "word_surface_benchmark.json").exists()


def test_run_word_surface_ab_benchmark_includes_delta(tmp_path):
    corpus = tmp_path / "corpus_delta.txt"
    corpus.write_text(
        "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. "
        "Sentence six. Sentence seven. Sentence eight. "
    )

    result = run_word_surface_ab_benchmark(
        corpus_path=str(corpus),
        prompts=["Tell me something useful."],
        prompt_sentences=1,
        target_sentences=1,
        examples=2,
        warmup_chars=60,
        response_steps=8,
        history_window=2,
        num_workers=1,
        use_dict=False,
        checkpoint_dir=str(tmp_path),
    )

    assert "delta" in result
    assert any(key.startswith("continuation_") for key in result["delta"])
    assert any(key.startswith("chat_") for key in result["delta"])
    assert any(key.startswith("child_") for key in result["delta"])


def test_run_word_surface_ab_benchmark_with_seed_libraries(tmp_path):
    corpus = tmp_path / "corpus_seed.txt"
    corpus.write_text(
        "The quick brown fox jumps over the lazy dog. "
        "The fox observes the dog. "
        "The dog wakes at dawn. "
        "The fox runs into the field. "
        "The fire goes out slowly. "
        "The dog rests again. "
        "Another sentence follows. "
    )

    word_base = tmp_path / "word_seed" / "word_seed_library"
    build_large_word_library(
        output=str(word_base),
        target=4,
        steps_per_chunk=80,
        min_density=0.0,
        keep_top_k=1,
        dedup_threshold=1.0,
        promote=True,
        num_workers=1,
        target_chars=5_000,
        registry_path=str(tmp_path / "registry.json"),
        name="word_seed_test",
        domain="text",
        corpus_paths=[str(corpus)],
        max_vocab_size=32,
        min_freq=1,
    )

    ascii_library = Path("library_bootstrap/nltk_large/nltk_large_nlp_2000.pkl")
    result = run_word_surface_ab_benchmark(
        corpus_path=str(corpus),
        prompts=[
            "Why is the model repeating itself?",
            "What changes when the corpus gets larger?",
        ],
        prompt_sentences=1,
        target_sentences=1,
        examples=2,
        warmup_chars=80,
        response_steps=10,
        history_window=3,
        num_workers=1,
        use_dict=False,
        checkpoint_dir=str(tmp_path),
        word_library_path=str(word_base),
        ascii_library_path=str(ascii_library) if ascii_library.exists() else None,
    )

    assert result["word"]["surface_mode"] == "word"
    assert result["ascii"]["surface_mode"] == "ascii"
    assert result["word"]["continuation"]["metrics"]["avg_token_agreement"] >= 0.0
    assert result["ascii"]["continuation"]["metrics"]["avg_token_agreement"] >= 0.0
