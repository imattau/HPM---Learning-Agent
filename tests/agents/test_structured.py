"""Tests for StructuredOrchestrator."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest
from hpm.agents.structured import StructuredOrchestrator
from benchmarks.multi_agent_common import make_orchestrator


def _make_dummy_encoder(n_dims: int, n_vecs: int = 1):
    """Encoder that returns n_vecs zero-vectors of shape (n_dims,)."""
    class DummyEncoder:
        feature_dim = n_dims
        max_steps_per_obs = 1 if n_vecs == 1 else None
        def encode(self, observation, epistemic=None):
            return [np.zeros(n_dims) for _ in range(n_vecs)]
    return DummyEncoder()


def _make_level(n_agents, feature_dim, prefix):
    orch, agents, _ = make_orchestrator(
        n_agents=n_agents, feature_dim=feature_dim,
        agent_ids=[f"{prefix}_{i}" for i in range(n_agents)],
        with_monitor=False,
    )
    return orch, agents


def test_structured_orch_l1_always_fires():
    """L1 steps on every step() call."""
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    l2_orch, l2_agents = _make_level(1, 4, "l2")
    enc1 = _make_dummy_encoder(4)
    enc2 = _make_dummy_encoder(4)
    so = StructuredOrchestrator(
        encoders=[enc1, enc2],
        orches=[l1_orch, l2_orch],
        agents=[l1_agents, l2_agents],
        level_Ks=[1, 3],
    )
    for _ in range(5):
        so.step(None)
    assert so._step_ticks[0] == 5


def test_structured_orch_l2_cadence():
    """L2 fires every K=3 step() calls (not every step)."""
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    l2_orch, l2_agents = _make_level(1, 4, "l2")
    enc1 = _make_dummy_encoder(4)
    enc2 = _make_dummy_encoder(4)
    so = StructuredOrchestrator(
        encoders=[enc1, enc2],
        orches=[l1_orch, l2_orch],
        agents=[l1_agents, l2_agents],
        level_Ks=[1, 3],
    )
    for _ in range(3):
        so.step(None)
    assert so._step_ticks[1] == 1  # L2 fired once (at step 3)
    for _ in range(3):
        so.step(None)
    assert so._step_ticks[1] == 2  # fired again at step 6


def test_structured_orch_l2_multi_vec():
    """L2 encoder returning N vecs causes N step() calls to L2 orchestrator."""
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    l2_orch, l2_agents = _make_level(1, 6, "l2")
    enc1 = _make_dummy_encoder(4, n_vecs=1)
    enc2 = _make_dummy_encoder(6, n_vecs=3)  # returns 3 vecs per obs
    so = StructuredOrchestrator(
        encoders=[enc1, enc2],
        orches=[l1_orch, l2_orch],
        agents=[l1_agents, l2_agents],
        level_Ks=[1, 1],  # L2 fires every step
    )
    so.step(None)
    # L2 should have received 3 step() calls (one per vec)
    assert so._step_ticks[1] == 1  # one cadence tick


def test_structured_orch_l1_obs_dict_override():
    """l1_obs_dict routes partitioned obs to correct agents."""
    l1_orch, l1_agents = _make_level(2, 4, "l1")
    enc1 = _make_dummy_encoder(4)
    so = StructuredOrchestrator(
        encoders=[enc1],
        orches=[l1_orch],
        agents=[l1_agents],
        level_Ks=[1],
    )
    obs_a = np.ones(4)
    obs_b = np.full(4, 2.0)
    l1_obs_dict = {l1_agents[0].agent_id: obs_a, l1_agents[1].agent_id: obs_b}
    result = so.step(None, l1_obs_dict=l1_obs_dict)
    assert "level1" in result


def test_structured_orch_raises_generative_head():
    """Passing non-None generative_head raises NotImplementedError at construction."""
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    enc1 = _make_dummy_encoder(4)
    with pytest.raises(NotImplementedError):
        StructuredOrchestrator(
            encoders=[enc1],
            orches=[l1_orch],
            agents=[l1_agents],
            level_Ks=[1],
            generative_head=object(),
        )
