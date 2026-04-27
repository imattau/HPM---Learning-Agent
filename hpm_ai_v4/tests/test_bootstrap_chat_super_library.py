from hpm_ai_v4.simulations.bootstrap_chat_super_library import merge_chat_libraries


def test_merge_chat_libraries_reids_patterns(tmp_path):
    from hpm_ai_v4.tools.serializer import PatternSerializer
    from hpm_ai_v4.pattern import FlatPattern

    lib1 = tmp_path / "one.pkl"
    lib2 = tmp_path / "two.pkl"
    PatternSerializer.save([FlatPattern.flat(0, obs_dim=5), FlatPattern.flat(1, obs_dim=5)], str(lib1))
    PatternSerializer.save([FlatPattern.flat(0, obs_dim=5)], str(lib2))

    out = tmp_path / "merged.pkl"
    result = merge_chat_libraries(output_path=str(out), sources=[str(lib1), str(lib2)])

    merged = PatternSerializer.load(str(out))
    assert result.pattern_count == 3
    assert len(merged) == 3
    assert len({p.id for p in merged}) == 3
