from __future__ import annotations
from dataclasses import dataclass
import pathlib
import numpy as np
from hpm.agents.agent import Agent
from hpm.agents.multi_agent import MultiAgentOrchestrator


@dataclass
class LevelBundle:
    """Structured inter-level signal: belief + confidence from one Level 1 agent."""
    agent_id: str
    mu: np.ndarray        # shape (D,) — top pattern mean
    weight: float         # top pattern's store weight
    epistemic_loss: float # running epistemic loss for that pattern
    strategic_confidence: float = 1.0  # L5 metacognitive gating signal (default: full trust)


def extract_bundle(agent: Agent) -> LevelBundle:
    """Extract a structured bundle from an agent's current state.

    Reads the top-weighted pattern from the agent's store.
    If the store is empty (only possible with manually-cleared stores in tests),
    returns a zero bundle with maximum uncertainty (epistemic_loss=1.0).
    """
    feature_dim = agent.config.feature_dim
    records = agent.store.query(agent.agent_id)

    if not records:
        return LevelBundle(
            agent_id=agent.agent_id,
            mu=np.zeros(feature_dim),
            weight=0.0,
            epistemic_loss=1.0,
        )

    top_pattern, top_weight = max(records, key=lambda r: r[1])
    epistemic_loss = agent.epistemic._running_loss.get(top_pattern.id, 0.0)

    return LevelBundle(
        agent_id=agent.agent_id,
        mu=top_pattern.mu.copy(),
        weight=float(top_weight),
        epistemic_loss=float(epistemic_loss),
    )


def encode_bundle(bundle: LevelBundle) -> np.ndarray:
    """Concatenate [mu, weight, epistemic_loss] into a single observation vector.

    Output shape: (D + 2,) where D = len(bundle.mu).
    This becomes the raw observation fed to Level 2 agents.
    """
    return np.concatenate([bundle.mu, [bundle.weight, bundle.epistemic_loss]])


class HierarchicalOrchestrator:
    """Coordinates a 2-level abstraction stack.

    Level 1 agents process raw observations every step.
    Level 2 agents receive structured bundles from Level 1 every K steps.

    Bundle format: np.concatenate([mu, [weight, epistemic_loss]])
    shape: (l1_feature_dim + 2,)

    All Level 2 agents receive the same bundle per call.
    N Level 1 agents produce N separate step() calls to level2_orch per cadence tick.
    """

    def __init__(
        self,
        level1_orch: MultiAgentOrchestrator,
        level2_orch: MultiAgentOrchestrator,
        level1_agents: list,
        level2_agents: list,
        K: int = 1,
    ):
        self.level1_orch = level1_orch
        self.level2_orch = level2_orch
        self.level1_agents = level1_agents
        self.level2_agents = level2_agents
        self.K = K
        self._t = 0

    def step(self, obs: np.ndarray) -> dict:
        """Step the hierarchy.

        Always steps all Level 1 agents.
        Steps Level 2 only when self._t % K == 0 (after increment).
        Returns {"level1": ..., "level2": {} or last_l2_result, "t": self._t}.
        """
        l1_obs = {a.agent_id: obs for a in self.level1_agents}
        l1_result = self.level1_orch.step(l1_obs)

        self._t += 1
        l2_result = {}

        if self._t % self.K == 0:
            l2_agent_ids = [a.agent_id for a in self.level2_agents]
            for l1_agent in self.level1_agents:
                bundle = extract_bundle(l1_agent)
                encoded = encode_bundle(bundle)
                l2_obs = {aid: encoded for aid in l2_agent_ids}
                l2_result = self.level2_orch.step(l2_obs)

        return {"level1": l1_result, "level2": l2_result, "t": self._t}


def make_hierarchical_orchestrator(
    n_l1_agents: int,
    n_l2_agents: int,
    l1_feature_dim: int,
    K: int = 1,
    l1_pattern_type: str = "gaussian",
    l2_pattern_type: str = "gaussian",
    l1_agent_ids: list[str] | None = None,
    l2_agent_ids: list[str] | None = None,
) -> tuple:
    """Build a 2-level HierarchicalOrchestrator.

    Level 2 feature_dim is automatically set to l1_feature_dim + 2.
    This is the only supported construction path — do not construct
    HierarchicalOrchestrator directly with mismatched orchestrators.

    Returns: (HierarchicalOrchestrator, level1_agents, level2_agents)
    """
    import sys
    _repo_root = str(pathlib.Path(__file__).resolve().parent.parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from benchmarks.multi_agent_common import make_orchestrator  # noqa: E402

    l1_ids = l1_agent_ids or [f"l1_{i}" for i in range(n_l1_agents)]
    l2_ids = l2_agent_ids or [f"l2_{i}" for i in range(n_l2_agents)]

    l1_orch, l1_agents, _ = make_orchestrator(
        n_agents=n_l1_agents,
        feature_dim=l1_feature_dim,
        agent_ids=l1_ids,
        pattern_types=[l1_pattern_type] * n_l1_agents,
    )
    l2_orch, l2_agents, _ = make_orchestrator(
        n_agents=n_l2_agents,
        feature_dim=l1_feature_dim + 2,
        agent_ids=l2_ids,
        pattern_types=[l2_pattern_type] * n_l2_agents,
    )

    return (
        HierarchicalOrchestrator(l1_orch, l2_orch, l1_agents, l2_agents, K=K),
        l1_agents,
        l2_agents,
    )
