from hpm_ai_v4.simulations.build_large_word_library import build_large_word_library
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.library_registry import LibraryRegistry


def test_build_large_word_library_local_corpus(tmp_path):
    corpus = tmp_path / "word_corpus.txt"
    corpus.write_text(
        "The quick brown fox jumps over the lazy dog. "
        "The fox observes the dog. "
        "The dog wakes at dawn. "
        "The fox runs into the field. "
        "The fire goes out slowly. "
    )
    registry_path = tmp_path / "registry.json"
    output_base = tmp_path / "word_large" / "word_large_library"

    result = build_large_word_library(
        output=str(output_base),
        target=8,
        steps_per_chunk=120,
        min_density=0.0,
        keep_top_k=2,
        dedup_threshold=1.0,
        promote=True,
        num_workers=1,
        target_chars=10_000,
        registry_path=str(registry_path),
        name="word_large_seed_test",
        domain="text",
        corpus_paths=[str(corpus)],
        max_vocab_size=64,
        min_freq=1,
    )

    assert result.pattern_count > 0
    assert (tmp_path / "word_large" / "word_large_library.l1.pkl").exists()
    assert (tmp_path / "word_large" / "word_large_library.surface.json").exists()

    agent = LayeredAgent(num_workers=1)
    loaded = agent.load_bundle(str(output_base))
    assert loaded >= 1
    assert agent.surface_mode == "word"
    assert agent.l1.obs_dim >= 8

    registry = LibraryRegistry(str(registry_path))
    entry = registry.require("word_large_seed_test")
    assert entry.status == "promoted"
    assert entry.bundle_kind == "stacked"
    assert entry.domain == "text"
    assert entry.decoder_families == ["word"]


def test_build_large_word_library_writes_surface_vocab(tmp_path):
    corpus = tmp_path / "word_corpus_vocab.txt"
    corpus.write_text(
        "Alpha beta gamma. Alpha beta delta. Alpha epsilon zeta. "
        "The fox jumps again. The fox jumps again."
    )
    output_base = tmp_path / "word_large_vocab" / "word_large_library"

    build_large_word_library(
        output=str(output_base),
        target=4,
        steps_per_chunk=80,
        min_density=0.0,
        keep_top_k=1,
        dedup_threshold=1.0,
        promote=False,
        num_workers=1,
        target_chars=5_000,
        registry_path=None,
        corpus_paths=[str(corpus)],
        max_vocab_size=32,
        min_freq=1,
    )

    surface_path = tmp_path / "word_large_vocab" / "word_large_library.surface.json"
    assert surface_path.exists()
    data = surface_path.read_text()
    assert '"surface_mode": "word"' in data
    assert '"word_vocab"' in data
