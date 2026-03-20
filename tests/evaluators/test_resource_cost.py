# tests/evaluators/test_resource_cost.py
import numpy as np
import pytest
from unittest.mock import MagicMock
from hpm.evaluators.resource_cost import ResourceCostEvaluator
from hpm.patterns.gaussian import GaussianPattern


def _make_pattern(dim: int = 4) -> GaussianPattern:
    return GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))


def _set_pressure(ev: ResourceCostEvaluator, mem_percent: float, cpu_percent: float) -> None:
    """Inject a mock psutil into the evaluator to simulate system load without real OS calls."""
    mock = MagicMock()
    mock.virtual_memory.return_value.percent = mem_percent
    mock.cpu_percent.return_value = cpu_percent
    ev._psutil = mock


def test_pressure_zero_when_idle():
    ev = ResourceCostEvaluator()
    _set_pressure(ev, mem_percent=0.0, cpu_percent=0.0)
    assert ev.pressure() == pytest.approx(0.0)


def test_pressure_one_when_maxed():
    ev = ResourceCostEvaluator(w_mem=0.5, w_cpu=0.5)
    _set_pressure(ev, mem_percent=100.0, cpu_percent=100.0)
    assert ev.pressure() == pytest.approx(1.0)


def test_pressure_weighted():
    ev = ResourceCostEvaluator(w_mem=0.8, w_cpu=0.2)
    _set_pressure(ev, mem_percent=50.0, cpu_percent=100.0)
    # 0.8 * 0.5 + 0.2 * 1.0 = 0.6
    assert ev.pressure() == pytest.approx(0.6)


def test_evaluate_returns_negative_under_pressure():
    ev = ResourceCostEvaluator(lambda_cost=1.0)
    _set_pressure(ev, mem_percent=80.0, cpu_percent=80.0)
    assert ev.evaluate(_make_pattern()) < 0.0


def test_evaluate_zero_when_lambda_cost_zero():
    ev = ResourceCostEvaluator(lambda_cost=0.0)
    _set_pressure(ev, mem_percent=100.0, cpu_percent=100.0)
    assert ev.evaluate(_make_pattern()) == pytest.approx(0.0)


def test_evaluate_zero_when_idle():
    ev = ResourceCostEvaluator(lambda_cost=1.0)
    _set_pressure(ev, mem_percent=0.0, cpu_percent=0.0)
    assert ev.evaluate(_make_pattern()) == pytest.approx(0.0)


def test_complex_pattern_penalised_more_than_simple():
    """Higher description_length -> more negative E_cost under same pressure."""
    ev = ResourceCostEvaluator(lambda_cost=1.0)
    _set_pressure(ev, mem_percent=80.0, cpu_percent=80.0)
    simple = GaussianPattern(mu=np.zeros(2), sigma=np.eye(2))
    complex_ = GaussianPattern(mu=np.zeros(16), sigma=np.eye(16))
    assert ev.evaluate(complex_) < ev.evaluate(simple)


from hpm.config import AgentConfig
from hpm.agents.agent import Agent


def test_agent_unaffected_when_delta_cost_zero():
    """delta_cost=0.0 (default): e_cost_mean present but contributes nothing to weights."""
    config = AgentConfig(agent_id="test", feature_dim=4, delta_cost=0.0)
    agent = Agent(config)
    result = agent.step(np.zeros(4))
    assert 'e_cost_mean' in result
    assert result['e_cost_mean'] == pytest.approx(0.0)


def test_agent_step_includes_e_cost_mean():
    """Agent with delta_cost > 0 returns negative e_cost_mean under high pressure."""
    config = AgentConfig(agent_id="test", feature_dim=4, delta_cost=1.0, lambda_cost=1.0)
    agent = Agent(config)
    _set_pressure(agent.resource_cost, mem_percent=80.0, cpu_percent=80.0)
    result = agent.step(np.zeros(4))
    assert 'e_cost_mean' in result
    assert result['e_cost_mean'] < 0.0


def test_high_pressure_reduces_total_vs_no_pressure():
    """Under high resource pressure with delta_cost > 0, e_cost_mean is more negative than under low pressure."""
    config = AgentConfig(agent_id="test", feature_dim=4, delta_cost=2.0, lambda_cost=1.0)

    agent_high = Agent(config)
    _set_pressure(agent_high.resource_cost, mem_percent=95.0, cpu_percent=95.0)
    result_high = agent_high.step(np.zeros(4))

    agent_low = Agent(config)
    _set_pressure(agent_low.resource_cost, mem_percent=5.0, cpu_percent=5.0)
    result_low = agent_low.step(np.zeros(4))

    assert result_high['e_cost_mean'] < result_low['e_cost_mean']
