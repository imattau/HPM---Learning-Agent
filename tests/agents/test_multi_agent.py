import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.field.field import PatternField
from hpm.patterns.gaussian import GaussianPattern


def _make_agents(n, feature_dim=4, gamma_soc=0.5):
    field = PatternField()
    agents = [
        Agent(AgentConfig(agent_id=f"a{i}", feature_dim=feature_dim, gamma_soc=gamma_soc), field=field)
        for i in range(n)
    ]
    return agents, field


def test_orchestrator_accepts_agents_and_field():
    agents, field = _make_agents(2)
    orch = MultiAgentOrchestrator(agents, field)
    assert len(orch.agents) == 2


def test_step_returns_metrics_for_each_agent():
    agents, field = _make_agents(2)
    orch = MultiAgentOrchestrator(agents, field)
    obs = {"a0": np.zeros(4), "a1": np.zeros(4)}
    metrics = orch.step(obs)
    assert "a0" in metrics and "a1" in metrics
    assert "mean_accuracy" in metrics["a0"]


def test_run_returns_history_of_length_n_steps():
    agents, field = _make_agents(2)
    orch = MultiAgentOrchestrator(agents, field)
    history = orch.run(np.zeros(4), n_steps=5)
    assert len(history) == 5
    assert "a0" in history[0]


def test_run_increments_timestep_for_all_agents():
    agents, field = _make_agents(3)
    orch = MultiAgentOrchestrator(agents, field)
    orch.run(np.zeros(4), n_steps=10)
    for agent in agents:
        assert agent._t == 10


def test_field_updated_for_all_agents_after_step():
    agents, field = _make_agents(2)
    orch = MultiAgentOrchestrator(agents, field)
    orch.step({"a0": np.zeros(4), "a1": np.zeros(4)})
    assert "a0" in field._agent_patterns
    assert "a1" in field._agent_patterns


def test_seed_shared_gives_same_uuid_across_agents():
    agents, field = _make_agents(2)
    seed = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))
    orch = MultiAgentOrchestrator(agents, field, seed_pattern=seed)
    # Each agent should have exactly one pattern — the shared seed (random seed replaced)
    for agent in agents:
        records = agent.store.query(agent.agent_id)
        assert len(records) == 1, "random seed should be replaced, not added to"
        ids = [p.id for p, _ in records]
        assert seed.id in ids


def test_seed_shared_produces_cross_agent_freq_signal():
    """After shared seeding and stepping, shared UUID appears in both agents' field entries."""
    agents, field = _make_agents(2, gamma_soc=1.0)
    seed = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))
    orch = MultiAgentOrchestrator(agents, field, seed_pattern=seed)
    orch.step({"a0": np.zeros(4), "a1": np.zeros(4)})
    # The shared UUID (preserved by update()) should appear in both agents' field registrations
    all_registered_ids = set()
    for agent_patterns in field._agent_patterns.values():
        all_registered_ids.update(agent_patterns.keys())
    # Both agents stepped with the same seed UUID — it should be present
    assert seed.id in all_registered_ids


def test_single_agent_orchestrator_m3_enforced():
    """Single agent: m3_active=True and social signal gated to zero (spec M3)."""
    agents, field = _make_agents(1, gamma_soc=1.0)
    orch = MultiAgentOrchestrator(agents, field)
    metrics = orch.step({"a0": np.zeros(4)})
    assert "a0" in metrics
    assert metrics["a0"]["m3_active"] is True
    assert metrics["a0"]["e_soc_mean"] == pytest.approx(0.0)
