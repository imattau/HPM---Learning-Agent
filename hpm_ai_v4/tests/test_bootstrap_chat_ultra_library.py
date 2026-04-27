from hpm_ai_v4.simulations.bootstrap_chat_ultra_library import bootstrap_chat_ultra_library


def test_bootstrap_chat_ultra_library_merges_many_sources(tmp_path, monkeypatch):
    from hpm_ai_v4.tools.serializer import PatternSerializer
    from hpm_ai_v4.pattern import FlatPattern

    # Seed libraries
    seed_a = tmp_path / "seed_a.pkl"
    seed_b = tmp_path / "seed_b.pkl"
    PatternSerializer.save([FlatPattern.flat(i, obs_dim=5) for i in range(4)], str(seed_a))
    PatternSerializer.save([FlatPattern.flat(i, obs_dim=5) for i in range(5)], str(seed_b))

    # Docs chunks will be created from these inputs.
    docs = []
    for idx in range(9):
        path = tmp_path / f"doc_{idx}.md"
        path.write_text(f"# Doc {idx}\nSome content {idx}\n")
        docs.append(str(path))

    monkeypatch.setattr(
        "hpm_ai_v4.simulations.bootstrap_chat_ultra_library._seed_libraries",
        lambda: [str(seed_a), str(seed_b)],
    )
    monkeypatch.setattr(
        "hpm_ai_v4.simulations.bootstrap_chat_ultra_library._docs_sources",
        lambda: docs,
    )
    monkeypatch.setattr(
        "hpm_ai_v4.simulations.bootstrap_chat_ultra_library.build_library",
        lambda **kwargs: PatternSerializer.save([FlatPattern.flat(0, obs_dim=5)], kwargs["output"]) or 0,
    )

    result = bootstrap_chat_ultra_library(output_dir=str(tmp_path / "ultra"))

    assert result.pattern_count == 12
    assert result.library_path.endswith("chat_ultra_library.pkl")
