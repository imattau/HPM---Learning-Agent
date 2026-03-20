import numpy as np
import pytest
from hpm.agents.agent import Agent
from hpm.config import AgentConfig
from hpm.domains.concept import ConceptLearningDomain, Concept
from hpm.metrics.hpm_predictions import sensitivity_ratio, curiosity_complexity_profile


@pytest.fixture
def trained_agent():
    cfg = AgentConfig(agent_id="metric_agent", feature_dim=4, eta=0.05, lambda_L=0.3)
    agent = Agent(cfg)
    domain = ConceptLearningDomain(
        concepts=[
            Concept(np.array([1.0, 0.0]), [np.array([0.5, 0.5])], label=0),
            Concept(np.array([0.0, 1.0]), [np.array([0.2, 0.8])], label=1),
        ],
        noise=0.05,
        seed=0,
    )
    for _ in range(100):
        agent.step(domain.observe())
    return agent, domain


def test_sensitivity_ratio_returns_float(trained_agent):
    agent, domain = trained_agent
    ratio = sensitivity_ratio(agent, domain, n_steps=20)
    assert isinstance(ratio, float)


def test_curiosity_profile_returns_dict(trained_agent):
    agent, domain = trained_agent
    # Domains at three complexity levels
    domains = {
        2.0: ConceptLearningDomain(
            [Concept(np.array([1.0, 0.0]), [np.array([0.5, 0.5])], label=0)],
            noise=0.01, seed=1
        ),
        10.0: ConceptLearningDomain(
            [Concept(np.array([1.0, 0.0]), [np.array([0.5, 0.5])], label=0)],
            noise=0.3, seed=2
        ),
        50.0: ConceptLearningDomain(
            [Concept(np.array([1.0, 0.0]), [np.array([0.5, 0.5])], label=0)],
            noise=2.0, seed=3
        ),
    }
    profile = curiosity_complexity_profile(agent, domains, n_steps=20)
    assert set(profile.keys()) == set(domains.keys())
    assert all(isinstance(v, float) for v in profile.values())
