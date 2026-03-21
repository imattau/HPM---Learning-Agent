import math
import json
import numpy as np
import pytest
from hpm.store.sqlite import SQLiteStore
from hpm.patterns.gaussian import GaussianPattern
from hpm.monitor.structural_law import StructuralLawMonitor


def _make_pattern(level: int, dim: int = 4) -> GaussianPattern:
    p = GaussianPattern(mu=np.random.randn(dim), sigma=np.eye(dim))
    p.level = level
    return p


def _seed_store(store, agent_id, levels):
    """Save one pattern per level value into the store."""
    n = len(levels)
    for i, lvl in enumerate(levels):
        p = _make_pattern(lvl)
        store.save(p, 1.0 / n, agent_id)


def test_light_metrics_every_step(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed_store(store, "a1", [1, 2, 4, 4, 5])
    monitor = StructuralLawMonitor(store, T_monitor=100)

    result = monitor.step(step_t=1, agents=[], total_conflict=0.1)

    assert result["pattern_count"] == 5
    assert result["level_distribution"][4] == 2
    assert result["level_distribution"][5] == 1
    assert result["level4plus_count"] == 3
    assert 0.0 <= result["level4plus_mean_weight"] <= 1.0
    assert result["conflict"] == pytest.approx(0.1)
    assert 0.0 <= result["stability_mean"] <= 1.0


def test_heavy_metrics_none_before_cadence(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed_store(store, "a1", [4, 5])
    monitor = StructuralLawMonitor(store, T_monitor=10)

    result = monitor.step(step_t=1, agents=[], total_conflict=0.0)

    assert result["diversity"] is None
    assert result["redundancy"] is None


def test_heavy_metrics_present_at_cadence(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed_store(store, "a1", [4, 5])
    monitor = StructuralLawMonitor(store, T_monitor=1)

    result = monitor.step(step_t=1, agents=[], total_conflict=0.0)

    assert result["diversity"] is not None
    assert result["redundancy"] is not None
    assert result["diversity"] >= 0.0
    assert 0.0 <= result["redundancy"] <= 1.0


def test_stability_mean_in_range(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed_store(store, "a1", [1, 2, 3, 4, 5])
    monitor = StructuralLawMonitor(store, T_monitor=100)
    result = monitor.step(step_t=1, agents=[], total_conflict=0.0)
    assert 0.0 <= result["stability_mean"] <= 1.0


def test_redundancy_zero_with_one_l4_pattern(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed_store(store, "a1", [4])  # only one Level 4+ pattern
    monitor = StructuralLawMonitor(store, T_monitor=1)
    result = monitor.step(step_t=1, agents=[], total_conflict=0.0)
    assert result["redundancy"] == pytest.approx(0.0)


def test_no_log_when_log_path_none(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed_store(store, "a1", [4])
    monitor = StructuralLawMonitor(store, T_monitor=1, log_path=None)
    monitor.step(step_t=1, agents=[], total_conflict=0.0)
    # No files created other than the DB
    files = list(tmp_path.iterdir())
    assert all(f.suffix == ".db" for f in files)


def test_json_log_appends(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed_store(store, "a1", [4, 5])
    log_path = str(tmp_path / "monitor.jsonl")
    monitor = StructuralLawMonitor(store, T_monitor=1, log_path=log_path)
    monitor.step(step_t=1, agents=[], total_conflict=0.0)
    monitor.step(step_t=2, agents=[], total_conflict=0.0)
    with open(log_path) as f:
        lines = [l for l in f.readlines() if l.strip()]
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert "step" in entry
    assert "diversity" in entry


def test_diversity_zero_single_pattern(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.db"))
    p = _make_pattern(1)
    store.save(p, 1.0, "a1")
    monitor = StructuralLawMonitor(store, T_monitor=1)
    result = monitor.step(step_t=1, agents=[], total_conflict=0.0)
    # Single pattern weight=1.0 → entropy = -1.0 * log(1.0) = 0.0
    assert result["diversity"] == pytest.approx(0.0, abs=1e-9)


# Integration tests with MultiAgentOrchestrator
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.agents.agent import Agent
from hpm.config import AgentConfig
from hpm.field.field import PatternField


def _make_agent(store, dim=4):
    agent_id = f"agent_{id(store)}"
    config = AgentConfig(feature_dim=dim, agent_id=agent_id)
    return Agent(config, store=store)


def test_monitor_none_returns_empty_field_quality(tmp_path):
    """Orchestrator with monitor=None returns empty field_quality."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = _make_agent(store)
    field = PatternField()
    orch = MultiAgentOrchestrator([agent], field=field, monitor=None)
    obs = {agent.agent_id: np.zeros(4)}
    result = orch.step(obs)
    assert result.get("field_quality") == {}


def test_monitor_integrated_returns_light_metrics(tmp_path):
    """Orchestrator with monitor returns field_quality with light metrics."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = _make_agent(store)
    field = PatternField()
    monitor = StructuralLawMonitor(store, T_monitor=100)
    orch = MultiAgentOrchestrator([agent], field=field, monitor=monitor)
    obs = {agent.agent_id: np.zeros(4)}
    result = orch.step(obs)
    fq = result["field_quality"]
    assert "pattern_count" in fq
    assert "level_distribution" in fq
    assert fq["pattern_count"] >= 0
