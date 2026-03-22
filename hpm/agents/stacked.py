from __future__ import annotations
from dataclasses import dataclass
import pathlib
import numpy as np

from hpm.agents.agent import Agent
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.agents.hierarchical import extract_bundle, encode_bundle


@dataclass
class LevelConfig:
    """Configuration for one level in a StackedOrchestrator.

    n_agents: number of agents at this level
    pattern_type: pattern substrate ("gaussian", "laplace", etc.)
    K: cadence — how many times the level below must fire before this level fires.
       Ignored for level 0 (L1 always steps on every call).
    agent_ids: optional explicit IDs; auto-generated as "l{level+1}_{j}" if None.
    """
    n_agents: int
    pattern_type: str = "gaussian"
    K: int = 1
    agent_ids: list[str] | None = None


class StackedOrchestrator:
    """N-level abstraction stack.

    Level 0 (L1) processes raw observations every step.
    Level i fires every K[i] fires of level i-1 (cadence relative to level below).
    Each level receives N_prev separate step() calls per cadence tick (witness model).

    Attributes:
        level_orches: one MultiAgentOrchestrator per level
        level_agents: one list of Agent per level — public, for external inspection
        level_Ks: level_Ks[0]=1 (unused); level_Ks[i]=cadence for level i (i>=1)
        _level_ticks: how many times each level has fired; initialised to [0]*n_levels
    """

    def __init__(
        self,
        level_orches: list[MultiAgentOrchestrator],
        level_agents: list[list[Agent]],
        level_Ks: list[int],
    ):
        self.level_orches = level_orches
        self.level_agents = level_agents
        self.level_Ks = level_Ks
        self._level_ticks: list[int] = [0] * len(level_orches)

    def step(self, obs: np.ndarray) -> dict:
        """Step the hierarchy on one raw observation.

        Increment-then-check order (same as HierarchicalOrchestrator):
        each level's tick counter is incremented AFTER that level fires;
        the cadence check for level i+1 uses the updated _level_ticks[i]
        from the same step() call. First L2 cadence fires when _level_ticks[0]==K.

        Returns dict with keys "level1".."level{n}" and "t" (_level_ticks[0]).
        Deeper levels return {} on non-cadence steps.
        """
        n = len(self.level_orches)
        results: list[dict] = [{} for _ in range(n)]

        # Step level 0 (L1) — always fires
        l1_obs = {a.agent_id: obs for a in self.level_agents[0]}
        results[0] = self.level_orches[0].step(l1_obs)
        self._level_ticks[0] += 1

        # Step higher levels on cadence.
        # fired[i] tracks whether level i fired this step (used to gate level i+1).
        fired = [True] + [False] * (n - 1)  # level 0 always fires
        for i in range(1, n):
            # Only fire if level i-1 fired this step AND its tick count is a
            # multiple of the cadence for level i.
            if fired[i - 1] and self._level_ticks[i - 1] % self.level_Ks[i] == 0:
                li_agent_ids = [a.agent_id for a in self.level_agents[i]]
                for prev_agent in self.level_agents[i - 1]:
                    bundle = extract_bundle(prev_agent)
                    encoded = encode_bundle(bundle)
                    # Witness model: all level-i agents receive the same bundle per call.
                    # N level-(i-1) agents → N separate step() calls to level_orches[i].
                    obs_dict = {aid: encoded for aid in li_agent_ids}
                    results[i] = self.level_orches[i].step(obs_dict)
                self._level_ticks[i] += 1
                fired[i] = True

        out = {f"level{i + 1}": results[i] for i in range(n)}
        out["t"] = self._level_ticks[0]
        return out


def make_stacked_orchestrator(
    l1_feature_dim: int,
    level_configs: list[LevelConfig],
) -> tuple[StackedOrchestrator, list[list[Agent]]]:
    """Build an N-level StackedOrchestrator from a list of LevelConfig objects.

    Feature dimensions computed automatically:
      level i has feature_dim = l1_feature_dim + 2*i (0-indexed)
      → L1=D, L2=D+2, L3=D+4, ...

    NOTE: level_configs[0].K is ignored. L1 always steps on every call.

    Returns: (StackedOrchestrator, list_of_agent_lists_per_level)
    Second element is a convenience alias for orch.level_agents (same objects).
    """
    import sys
    _repo_root = str(pathlib.Path(__file__).resolve().parent.parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from benchmarks.multi_agent_common import make_orchestrator  # noqa: E402

    level_orches = []
    all_agents = []

    for i, cfg in enumerate(level_configs):
        feature_dim = l1_feature_dim + 2 * i
        agent_ids = cfg.agent_ids or [f"l{i + 1}_{j}" for j in range(cfg.n_agents)]
        orch, agents, _ = make_orchestrator(
            n_agents=cfg.n_agents,
            feature_dim=feature_dim,
            agent_ids=agent_ids,
            pattern_types=[cfg.pattern_type] * cfg.n_agents,
        )
        level_orches.append(orch)
        all_agents.append(agents)

    level_Ks = [1] + [cfg.K for cfg in level_configs[1:]]
    stacked = StackedOrchestrator(level_orches, all_agents, level_Ks)
    return stacked, all_agents
