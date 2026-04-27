from hpm_ai_v4.simulations.bootstrap_chat_ultra_bundle import bootstrap_chat_ultra_bundle


def test_bootstrap_chat_ultra_bundle_partitions_levels(tmp_path):
    from hpm_ai_v4.pattern import FlatPattern
    from hpm_ai_v4.tools.serializer import PatternSerializer

    source = tmp_path / "ultra.pkl"
    patterns = []
    patterns.extend([FlatPattern.flat(i, obs_dim=5) for i in range(6)])
    patterns.extend([FlatPattern.flat(100 + i, obs_dim=2) for i in range(6)])
    patterns.extend([FlatPattern.flat(200 + i, obs_dim=32) for i in range(2)])
    patterns.extend([FlatPattern.flat(300 + i, obs_dim=64) for i in range(2)])
    PatternSerializer.save(patterns, str(source))

    result = bootstrap_chat_ultra_bundle(
        output_dir=str(tmp_path / "bundle"),
        registry_path=str(tmp_path / "registry.json"),
        source_library=str(source),
    )

    assert result.pattern_count == 16
    assert result.level_counts["l1"] == 6
    assert result.level_counts["l2"] == 3
    assert result.level_counts["l3"] == 3
    assert result.level_counts["l4"] == 2
    assert result.level_counts["l5"] == 2
    assert (tmp_path / "bundle" / "chat_ultra_bundle.l1.pkl").exists()
    assert (tmp_path / "bundle" / "chat_ultra_bundle.l2.pkl").exists()
    assert (tmp_path / "bundle" / "chat_ultra_bundle.l3.pkl").exists()
    assert (tmp_path / "bundle" / "chat_ultra_bundle.l4.pkl").exists()
    assert (tmp_path / "bundle" / "chat_ultra_bundle.l5.pkl").exists()
