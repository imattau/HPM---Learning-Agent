import numpy as np

from hpm_ai_v4.pattern import HierarchicalPattern
from hpm_ai_v4.simulations import build_large_nlp_library as large_lib
from hpm_ai_v4.tools.library_registry import LibraryRegistry


def test_build_large_library_registers_seed_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(large_lib, "load_hf_chunks", lambda target_chars=0: ["a" * 200, "b" * 200])

    def fake_train_chunk(args):
        chunk_text, steps, min_density, chunk_idx, keep_top_k = args
        patterns = []
        for offset in range(2):
            pattern = HierarchicalPattern(pattern_id=chunk_idx * 10 + offset, latent_dim=2, obs_dim=5)
            pattern.weight = 0.5 + 0.1 * offset
            pattern.source_corpus = f"chunk_{chunk_idx}"
            pattern.density_at_save = 0.35 + 0.05 * offset
            patterns.append(pattern)
        return patterns

    monkeypatch.setattr(large_lib, "_train_chunk", fake_train_chunk)
    np.random.seed(7)

    output = tmp_path / "large_nlp.pkl"
    registry = tmp_path / "registry.json"
    result = large_lib.build_large_library(
        output=str(output),
        target=3,
        steps_per_chunk=10,
        min_density=0.0,
        num_workers=1,
        target_chars=1000,
        registry_path=str(registry),
        name="large_nlp_seed",
        domain="text",
    )

    assert result.pattern_count >= 3
    assert result.chunk_count == 2
    assert output.exists()
    assert registry.exists()

    entry = LibraryRegistry(str(registry)).require("large_nlp_seed")
    assert entry.domain == "text"
    assert entry.status == "seed"
    assert entry.bundle_kind == "flat"
    assert entry.level_contract == "l1"
    assert entry.pattern_count == result.pattern_count


def test_build_large_library_can_promote_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(large_lib, "load_hf_chunks", lambda target_chars=0: ["a" * 200, "b" * 200])
    monkeypatch.setattr(large_lib, "_train_chunk", lambda args: [])

    output = tmp_path / "large_nlp.pkl"
    registry = tmp_path / "registry.json"
    result = large_lib.build_large_library(
        output=str(output),
        target=1,
        steps_per_chunk=10,
        min_density=0.0,
        keep_top_k=1,
        dedup_threshold=1.0,
        promote=True,
        num_workers=1,
        target_chars=1000,
        registry_path=str(registry),
        name="large_nlp_seed",
        domain="text",
    )

    assert result.pattern_count == 0
    assert registry.exists()

    entry = LibraryRegistry(str(registry)).require("large_nlp_seed")
    assert entry.status == "promoted"
