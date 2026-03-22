import numpy as np
import pytest
from hpm.agents.hierarchical import LevelBundle, encode_bundle


def test_encode_bundle_shape():
    bundle = LevelBundle(agent_id="a", mu=np.zeros(16), weight=0.5, epistemic_loss=0.1)
    encoded = encode_bundle(bundle)
    assert encoded.shape == (18,)


def test_encode_bundle_values():
    mu = np.ones(4)
    bundle = LevelBundle(agent_id="a", mu=mu, weight=0.7, epistemic_loss=0.3)
    encoded = encode_bundle(bundle)
    np.testing.assert_allclose(encoded[:4], mu)
    assert encoded[4] == pytest.approx(0.7)
    assert encoded[5] == pytest.approx(0.3)


def test_level_bundle_fields():
    b = LevelBundle(agent_id="x", mu=np.zeros(8), weight=1.0, epistemic_loss=0.0)
    assert b.agent_id == "x"
    assert b.mu.shape == (8,)
    assert b.weight == 1.0
    assert b.epistemic_loss == 0.0
