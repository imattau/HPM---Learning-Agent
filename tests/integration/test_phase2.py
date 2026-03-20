import numpy as np
import pytest
from hpm.agents.agent import Agent
from hpm.config import AgentConfig
from hpm.domains.concept import ConceptLearningDomain, Concept
from hpm.store.sqlite import SQLiteStore
from hpm.substrate.local_file import LocalFileSubstrate


def make_domain(seed=0):
    return ConceptLearningDomain(
        concepts=[
            Concept(np.array([1.0, 0.0]), [np.array([0.5, 0.5])], label=0),
            Concept(np.array([0.0, 1.0]), [np.array([0.2, 0.8])], label=1),
        ],
        noise=0.05,
        seed=seed,
    )


def test_agent_resumes_across_sessions(tmp_path):
    """Patterns trained in session 1 are present in session 2 via SQLiteStore."""
    db = str(tmp_path / "agent.db")
    cfg = AgentConfig(agent_id="persist_agent", feature_dim=4)
    domain = make_domain()

    # Session 1: train for 30 steps
    agent1 = Agent(cfg, store=SQLiteStore(db))
    for _ in range(30):
        agent1.step(domain.observe())
    ids1 = {p.id for p, _ in agent1.store.query("persist_agent")}
    assert len(ids1) >= 1

    # Session 2: new Agent instance on same DB — patterns from session 1 present
    agent2 = Agent(cfg, store=SQLiteStore(db))
    ids2 = {p.id for p, _ in agent2.store.query("persist_agent")}
    assert ids1 == ids2


def test_agent_with_substrate_does_not_error(tmp_path):
    """Agent.step() runs without error when substrate is attached."""
    text_dir = tmp_path / "texts"
    text_dir.mkdir()
    (text_dir / "a.txt").write_text("concept one definition here")
    (text_dir / "b.txt").write_text("concept two explanation there")

    substrate = LocalFileSubstrate(str(text_dir), feature_dim=4)
    cfg = AgentConfig(agent_id="substrate_agent", feature_dim=4)
    agent = Agent(cfg, substrate=substrate)
    domain = make_domain()

    for _ in range(10):
        result = agent.step(domain.observe())
    assert 'mean_accuracy' in result
    assert 'ext_field_freq' in result


def test_step_returns_ext_field_freq(tmp_path):
    """step() returns ext_field_freq key when substrate is set."""
    text_dir = tmp_path / "texts"
    text_dir.mkdir()
    (text_dir / "c.txt").write_text("learning patterns neural networks")

    substrate = LocalFileSubstrate(str(text_dir), feature_dim=4)
    cfg = AgentConfig(agent_id="freq_agent", feature_dim=4)
    agent = Agent(cfg, substrate=substrate)

    result = agent.step(np.zeros(4))
    assert 'ext_field_freq' in result
    assert 0.0 <= result['ext_field_freq'] <= 1.0


def test_step_without_substrate_has_zero_ext_field_freq():
    """Without substrate, ext_field_freq is 0.0."""
    cfg = AgentConfig(agent_id="no_sub_agent", feature_dim=4)
    agent = Agent(cfg)
    result = agent.step(np.zeros(4))
    assert result['ext_field_freq'] == pytest.approx(0.0)
