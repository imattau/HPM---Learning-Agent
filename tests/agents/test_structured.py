"""Tests for StructuredOrchestrator."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest
from hpm.agents.structured import StructuredOrchestrator
from hpm.agents.relational import RelationalEdge, StructuralMessage
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


def test_structured_orch_accepts_generative_head():
    """Passing non-None generative_head is now accepted (L4 implemented)."""
    from hpm.agents.l4_generative import L4GenerativeHead
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    enc1 = _make_dummy_encoder(4)
    head = L4GenerativeHead(feature_dim_in=4, feature_dim_out=4)
    so = StructuredOrchestrator(
        encoders=[enc1],
        orches=[l1_orch],
        agents=[l1_agents],
        level_Ks=[1],
        generative_head=head,
    )
    assert so.generative_head is head


def test_structured_orch_reset_clears_l4_l5():
    """reset() clears L4 and L5 state."""
    from hpm.agents.l4_generative import L4GenerativeHead
    from hpm.agents.l5_monitor import L5MetaMonitor
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    enc1 = _make_dummy_encoder(4)
    head = L4GenerativeHead(feature_dim_in=4, feature_dim_out=4)
    monitor = L5MetaMonitor()
    so = StructuredOrchestrator(
        encoders=[enc1],
        orches=[l1_orch],
        agents=[l1_agents],
        level_Ks=[1],
        generative_head=head,
        meta_monitor=monitor,
    )
    rng = np.random.default_rng(0)
    so.generative_head.accumulate(rng.standard_normal(4), rng.standard_normal(4))
    so.generative_head.accumulate(rng.standard_normal(4), rng.standard_normal(4))
    so.generative_head.fit()
    so.meta_monitor.update(rng.standard_normal(4), rng.standard_normal(4))
    so.reset()
    assert so.generative_head.predict(np.zeros(4)) is None
    assert so.meta_monitor.strategic_confidence() == 1.0


def test_l4_accumulate_and_update_adds_pairs():
    """_l4_accumulate_and_update stores pairs in L4 head."""
    from hpm.agents.l4_generative import L4GenerativeHead
    from hpm.agents.l5_monitor import L5MetaMonitor
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    enc1 = _make_dummy_encoder(4)
    head = L4GenerativeHead(feature_dim_in=4, feature_dim_out=4)
    monitor = L5MetaMonitor()
    so = StructuredOrchestrator(
        encoders=[enc1],
        orches=[l1_orch],
        agents=[l1_agents],
        level_Ks=[1],
        generative_head=head,
        meta_monitor=monitor,
    )
    rng = np.random.default_rng(1)
    for _ in range(3):
        so._l4_accumulate_and_update(rng.standard_normal(4), rng.standard_normal(4))
    assert len(so.generative_head._X) == 3


def test_structured_orch_passes_relational_bundles_to_supported_encoder():
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    l2_orch, l2_agents = _make_level(1, 6, "l2")
    enc1 = _make_dummy_encoder(4)

    class RelationalAwareEncoder:
        feature_dim = 6
        max_steps_per_obs = 1

        def __init__(self):
            self.calls = []

        def encode(self, observation, epistemic=None, relational_bundles=None):
            self.calls.append((observation, epistemic, relational_bundles))
            assert relational_bundles is not None
            assert len(relational_bundles) == 1
            return [np.zeros(6)]

    enc2 = RelationalAwareEncoder()
    so = StructuredOrchestrator(
        encoders=[enc1, enc2],
        orches=[l1_orch, l2_orch],
        agents=[l1_agents, l2_agents],
        level_Ks=[1, 1],
        relational_bundles_enabled=True,
    )

    so.step(np.zeros(4))
    assert len(enc2.calls) == 1
    _, epistemic, bundles = enc2.calls[0]
    assert epistemic is not None
    assert bundles[0].agent_id == l1_agents[0].agent_id
    assert len(bundles[0].relations) == 3


def test_structured_orch_ignores_relational_bundles_for_legacy_encoder():
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    l2_orch, l2_agents = _make_level(1, 6, "l2")
    enc1 = _make_dummy_encoder(4)

    class LegacyEncoder:
        feature_dim = 6
        max_steps_per_obs = 1

        def __init__(self):
            self.calls = []

        def encode(self, observation, epistemic=None):
            self.calls.append((observation, epistemic))
            return [np.zeros(6)]

    enc2 = LegacyEncoder()
    so = StructuredOrchestrator(
        encoders=[enc1, enc2],
        orches=[l1_orch, l2_orch],
        agents=[l1_agents, l2_agents],
        level_Ks=[1, 1],
        relational_bundles_enabled=True,
    )

    so.step(np.zeros(4))
    assert len(enc2.calls) == 1


def test_structured_orch_default_keeps_relational_bundles_disabled():
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    l2_orch, l2_agents = _make_level(1, 6, "l2")
    enc1 = _make_dummy_encoder(4)

    class RelationalAwareEncoder:
        feature_dim = 6
        max_steps_per_obs = 1

        def __init__(self):
            self.calls = []

        def encode(self, observation, epistemic=None, relational_bundles=None):
            self.calls.append(relational_bundles)
            return [np.zeros(6)]

    enc2 = RelationalAwareEncoder()
    so = StructuredOrchestrator(
        encoders=[enc1, enc2],
        orches=[l1_orch, l2_orch],
        agents=[l1_agents, l2_agents],
        level_Ks=[1, 1],
    )

    so.step(np.zeros(4))
    assert enc2.calls == [None]


def test_structured_orch_passes_structural_messages_to_supported_encoder():
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    l2_orch, l2_agents = _make_level(1, 6, "l2")
    enc1 = _make_dummy_encoder(4)

    class MessageAwareEncoder:
        feature_dim = 6
        max_steps_per_obs = 1

        def __init__(self):
            self.calls = []

        def encode(self, observation, epistemic=None, structural_messages=None):
            self.calls.append(structural_messages)
            assert structural_messages is not None
            return [np.zeros(6)]

    enc2 = MessageAwareEncoder()
    so = StructuredOrchestrator(
        encoders=[enc1, enc2],
        orches=[l1_orch, l2_orch],
        agents=[l1_agents, l2_agents],
        level_Ks=[1, 1],
        structural_messages_to_encoders_enabled=True,
    )

    msg = StructuralMessage(
        source_agent_id='sender',
        relations=(
            RelationalEdge(
                source='agent:sender',
                relation='tracks_pattern',
                target='pattern:fake',
                confidence=1.0,
            ),
        ),
        confidence=1.0,
        provenance=('agent:sender',),
    )
    l1_agents[0].accept_structural_message(msg, 'sender')

    so.step(np.zeros(4))
    assert len(enc2.calls) == 1
    assert len(enc2.calls[0]) == 1
    source_id, message = enc2.calls[0][0]
    assert source_id == 'sender'
    assert message is msg
    assert l1_agents[0].consume_structural_inbox(clear=False) == []


def test_structured_orch_ignores_structural_messages_for_legacy_encoder():
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    l2_orch, l2_agents = _make_level(1, 6, "l2")
    enc1 = _make_dummy_encoder(4)

    class LegacyEncoder:
        feature_dim = 6
        max_steps_per_obs = 1

        def __init__(self):
            self.calls = []

        def encode(self, observation, epistemic=None):
            self.calls.append((observation, epistemic))
            return [np.zeros(6)]

    enc2 = LegacyEncoder()
    so = StructuredOrchestrator(
        encoders=[enc1, enc2],
        orches=[l1_orch, l2_orch],
        agents=[l1_agents, l2_agents],
        level_Ks=[1, 1],
        structural_messages_to_encoders_enabled=True,
    )

    msg = StructuralMessage(source_agent_id='sender', relations=(), confidence=1.0, provenance=('agent:sender',))
    l1_agents[0].accept_structural_message(msg, 'sender')

    so.step(np.zeros(4))
    assert len(enc2.calls) == 1
    # Context is drained at orchestrator level even when encoder cannot consume it.
    assert l1_agents[0].consume_structural_inbox(clear=False) == []


def test_structured_orch_passes_identity_snapshots_to_supported_encoder():
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    l2_orch, l2_agents = _make_level(1, 6, "l2")
    enc1 = _make_dummy_encoder(4)

    class IdentityAwareEncoder:
        feature_dim = 6
        max_steps_per_obs = 1

        def __init__(self):
            self.calls = []

        def encode(self, observation, epistemic=None, identity_snapshots=None):
            self.calls.append(identity_snapshots)
            assert identity_snapshots is not None
            return [np.zeros(6)]

    enc2 = IdentityAwareEncoder()
    so = StructuredOrchestrator(
        encoders=[enc1, enc2],
        orches=[l1_orch, l2_orch],
        agents=[l1_agents, l2_agents],
        level_Ks=[1, 1],
        identity_snapshots_to_encoders_enabled=True,
    )

    so.step(np.zeros(4))
    assert len(enc2.calls) == 1
    assert isinstance(enc2.calls[0], list)
    assert enc2.calls[0][0] != {}
    snapshot = next(iter(enc2.calls[0][0].values()))
    assert "identity" in snapshot
    assert "state" in snapshot
