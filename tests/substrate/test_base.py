import numpy as np
import pytest
from hpm.substrate.base import hash_vectorise


def test_hash_vectorise_returns_correct_dim():
    vec = hash_vectorise("hello world", dim=16)
    assert vec.shape == (16,)


def test_hash_vectorise_is_normalised():
    vec = hash_vectorise("hello world test text", dim=16)
    assert abs(vec.sum() - 1.0) < 1e-6


def test_hash_vectorise_empty_text():
    vec = hash_vectorise("", dim=16)
    assert vec.shape == (16,)
    assert vec.sum() == pytest.approx(0.0)


def test_hash_vectorise_deterministic():
    v1 = hash_vectorise("the quick brown fox", dim=32)
    v2 = hash_vectorise("the quick brown fox", dim=32)
    assert np.allclose(v1, v2)


def test_config_has_alpha_int():
    from hpm.config import AgentConfig
    cfg = AgentConfig(agent_id="a", feature_dim=4)
    assert hasattr(cfg, 'alpha_int')
    assert cfg.alpha_int == pytest.approx(0.8)
