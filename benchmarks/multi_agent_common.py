"""
multi_agent_common.py — Shared utilities for multi-agent HPM benchmark scripts.

Provides:
  - make_orchestrator(): construct N agents sharing a store + field + monitor + strategist
  - avg_metric(): average a per-agent metric across all agents in a result dict
  - step_all(): step orchestrator with all agents seeing the same observation
  - compute_redundancy(): snapshot field redundancy without triggering console output
  - print_results_table(): re-exported from common
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.field.field import PatternField
from hpm.store.memory import InMemoryStore
from hpm.monitor.structural_law import StructuralLawMonitor
from hpm.monitor.recombination_strategist import RecombinationStrategist
from benchmarks.common import BENCH_CONFIG, print_results_table  # noqa: F401


def make_orchestrator(
    n_agents: int = 2,
    feature_dim: int = 16,
    agent_ids: list[str] | None = None,
    with_monitor: bool = True,
    with_strategist: bool = True,
    T_monitor: int = 50,
    **overrides,
) -> tuple:
    """
    Construct N agents sharing a single InMemoryStore and PatternField,
    with StructuralLawMonitor (verbose=False) and RecombinationStrategist.

    The monitor runs every T_monitor steps providing diversity/redundancy
    metrics to the strategist, which triggers recombination bursts to prevent
    convergence and encourage cross-agent learning.

    All agents share the same store so the monitor observes the full population.
    Each agent is namespaced by agent_id.

    Returns:
        (orch, agents, store)
    """
    if agent_ids is None:
        agent_ids = [f"agent_{i}" for i in range(n_agents)]

    store = InMemoryStore()
    field = PatternField()

    cfg_kwargs = dict(BENCH_CONFIG)
    cfg_kwargs["feature_dim"] = feature_dim
    cfg_kwargs.update(overrides)

    agents = [
        Agent(AgentConfig(agent_id=aid, **cfg_kwargs), store=store, field=field)
        for aid in agent_ids
    ]

    monitor = (
        StructuralLawMonitor(store, T_monitor=T_monitor, verbose=False)
        if with_monitor else None
    )
    strategist = (
        RecombinationStrategist()
        if (with_strategist and with_monitor) else None
    )
    orch = MultiAgentOrchestrator(agents, field, monitor=monitor, strategist=strategist)

    return orch, agents, store


def step_all(orch: MultiAgentOrchestrator, agents: list, obs: np.ndarray) -> dict:
    """Step orchestrator with all agents receiving the same observation."""
    observations = {a.agent_id: obs for a in agents}
    return orch.step(observations)


def avg_metric(result: dict, agents: list, key: str, default: float = 0.0) -> float:
    """Average a per-agent metric across all agents."""
    vals = [result[a.agent_id].get(key, default) for a in agents]
    return float(np.mean(vals))


def compute_redundancy(orch: MultiAgentOrchestrator) -> float | None:
    """
    Compute field redundancy snapshot without triggering monitor console output.

    Calls _compute_heavy() directly on the monitor's store snapshot.
    Returns None if monitor is not configured.
    """
    if orch.monitor is None:
        return None
    all_records = orch.monitor._store.query_all()
    patterns = [p for p, _, _ in all_records]
    weights = [w for _, w, _ in all_records]
    if not patterns:
        return None
    _, redundancy = orch.monitor._compute_heavy(patterns, weights)
    return redundancy
