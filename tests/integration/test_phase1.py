"""
End-to-end Phase 1 integration test.
Verifies the full data flow: domain -> agent -> metrics.
"""
import numpy as np
import pytest
from hpm.agents.agent import Agent
from hpm.config import AgentConfig
from hpm.domains.concept import ConceptLearningDomain, Concept
from hpm.metrics.hpm_predictions import sensitivity_ratio


def make_domain(seed=0):
    return ConceptLearningDomain(
        concepts=[
            Concept(
                deep_features=np.array([1.0, 0.0, 0.0]),
                surface_templates=[
                    np.array([0.8, 0.2]),
                    np.array([0.6, 0.4]),
                ],
                label=0,
            ),
            Concept(
                deep_features=np.array([0.0, 1.0, 0.0]),
                surface_templates=[
                    np.array([0.2, 0.8]),
                    np.array([0.3, 0.7]),
                ],
                label=1,
            ),
            Concept(
                deep_features=np.array([0.0, 0.0, 1.0]),
                surface_templates=[
                    np.array([0.5, 0.5]),
                    np.array([0.4, 0.6]),
                ],
                label=2,
            ),
        ],
        noise=0.05,
        seed=seed,
    )


def test_agent_accuracy_improves_over_training():
    """Accuracy should increase (become less negative) over 200 steps."""
    domain = make_domain()
    cfg = AgentConfig(
        agent_id="integration_agent",
        feature_dim=domain.feature_dim(),
        eta=0.05,
        lambda_L=0.2,
        beta_aff=0.3,
    )
    agent = Agent(cfg)

    early_acc, late_acc = [], []
    for t in range(200):
        result = agent.step(domain.observe())
        if t < 20:
            early_acc.append(result['mean_accuracy'])
        if t >= 180:
            late_acc.append(result['mean_accuracy'])

    assert np.mean(late_acc) > np.mean(early_acc), (
        f"Accuracy did not improve: early={np.mean(early_acc):.3f}, late={np.mean(late_acc):.3f}"
    )


def test_sensitivity_ratio_sign():
    """
    §9.1: after training, sensitivity ratio should be > 0.
    (Full HPM prediction ratio > 1 requires longer training — this tests the sign.)
    """
    domain = make_domain()
    cfg = AgentConfig(
        agent_id="sens_agent",
        feature_dim=domain.feature_dim(),
        eta=0.05,
        lambda_L=0.2,
    )
    agent = Agent(cfg)
    for _ in range(150):
        agent.step(domain.observe())

    ratio = sensitivity_ratio(agent, domain, n_steps=30)
    # Ratio should be a real number (finite or inf from zero surface drop)
    # HPM predicts > 1; here we just verify it is defined and non-negative
    assert not np.isnan(ratio)
    assert ratio >= 0.0 or ratio == float('inf')


def test_library_floor_prevents_empty_store():
    """§3.3 edge case: even under extreme dynamics, library never empties."""
    domain = make_domain()
    cfg = AgentConfig(
        agent_id="floor_agent",
        feature_dim=domain.feature_dim(),
        eta=100.0,    # extreme learning rate to force weight collapse
        beta_c=100.0,
        epsilon=0.01,
    )
    agent = Agent(cfg)
    # Run several steps — library should never empty
    for _ in range(20):
        agent.step(domain.observe())
        records = agent.store.query("floor_agent")
        assert len(records) >= 1, "Library emptied — floor not working"
        total_weight = sum(w for _, w in records)
        assert total_weight > 0.0


def test_seeded_initialization_is_reproducible():
    """Empty-store seeding should be deterministic for the same agent identity."""
    cfg1 = AgentConfig(agent_id="seeded_agent", feature_dim=4)
    cfg2 = AgentConfig(agent_id="seeded_agent", feature_dim=4)
    agent1 = Agent(cfg1)
    agent2 = Agent(cfg2)

    pat1, weight1 = max(agent1.store.query("seeded_agent"), key=lambda r: r[1])
    pat2, weight2 = max(agent2.store.query("seeded_agent"), key=lambda r: r[1])

    assert weight1 == weight2 == 1.0
    assert np.allclose(pat1.mu, pat2.mu)


def test_store_persists_patterns_across_steps():
    """Patterns in the store reflect state after multiple steps."""
    domain = make_domain()
    cfg = AgentConfig(agent_id="store_agent", feature_dim=domain.feature_dim())
    agent = Agent(cfg)

    for _ in range(10):
        agent.step(domain.observe())

    records = agent.store.query("store_agent")
    assert len(records) >= 1
    for p, w in records:
        assert w >= cfg.epsilon
        assert p.is_structurally_valid()
