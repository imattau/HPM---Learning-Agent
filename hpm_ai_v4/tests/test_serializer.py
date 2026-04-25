# hpm_ai_v4/tests/test_serializer.py
import tempfile, os
import numpy as np
import pytest
from hpm_ai_v4.pattern import HierarchicalPattern
from hpm_ai_v4.tools.serializer import PatternSerializer


def test_save_load_roundtrip(tmp_path):
    p = HierarchicalPattern(42, latent_dim=2, obs_dim=6)
    path = str(tmp_path / "test.pkl")
    PatternSerializer.save([p], path)
    loaded = PatternSerializer.load(path)
    assert len(loaded) == 1
    assert loaded[0].id == 42
    assert loaded[0].A.shape == (2, 2)
    assert np.allclose(loaded[0].A, p.A, atol=1e-5)


def test_load_preserves_weights(tmp_path):
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    p.weight = 0.75
    path = str(tmp_path / "w.pkl")
    PatternSerializer.save([p], path)
    loaded = PatternSerializer.load(path)
    assert abs(loaded[0].weight - 0.75) < 1e-5


def test_save_multiple(tmp_path):
    patterns = [HierarchicalPattern(i, latent_dim=2, obs_dim=6) for i in range(5)]
    path = str(tmp_path / "multi.pkl")
    PatternSerializer.save(patterns, path)
    loaded = PatternSerializer.load(path)
    assert len(loaded) == 5
    assert [p.id for p in loaded] == list(range(5))


def test_json_roundtrip(tmp_path):
    p = HierarchicalPattern(7, latent_dim=2, obs_dim=6)
    path = str(tmp_path / "test.json")
    PatternSerializer.save_json([p], path)
    loaded = PatternSerializer.load_json(path)
    assert loaded[0].id == 7
    assert np.allclose(loaded[0].B, p.B, atol=1e-5)


from hpm_ai_v4.agents.agent import HPMAgent


def test_load_library_replaces_patterns(tmp_path):
    patterns = [HierarchicalPattern(i, latent_dim=2, obs_dim=6) for i in range(3)]
    path = str(tmp_path / "lib.pkl")
    PatternSerializer.save(patterns, path)
    agent = HPMAgent()
    n = agent.load_library(path)
    assert n == 3
    assert len(agent.patterns) == 3


def test_load_library_resets_weights(tmp_path):
    patterns = [HierarchicalPattern(i, latent_dim=2, obs_dim=6) for i in range(4)]
    for p in patterns:
        p.weight = 0.99
    path = str(tmp_path / "lib2.pkl")
    PatternSerializer.save(patterns, path)
    agent = HPMAgent()
    agent.load_library(path, reset_weights=True)
    for p in agent.patterns:
        assert abs(p.weight - 0.25) < 1e-5
