"""StructuredOrchestrator: domain-agnostic multi-level HPM orchestrator.

Each level receives a domain-specific observation via its LevelEncoder instance.
Cadence for level i is based on total step() calls, not per-object ticks.
Epistemic state (weight, epistemic_loss) threads from each level into the next encoder.
"""
from __future__ import annotations
import numpy as np


class StructuredOrchestrator:
    """N-level HPM orchestrator with per-level LevelEncoder instances.

    Attributes:
        encoders: One LevelEncoder per level.
        orches: One MultiAgentOrchestrator per level.
        agents: One list[Agent] per level.
        level_Ks: Cadence per level. level_Ks[0] unused (L1 always fires).
                  level_Ks[i] = K means level i fires every K step() calls.
        _step_ticks: How many times step() has been called (index 0 = total).
                     _step_ticks[i] counts how many times level i has fired.
    """

    def __init__(
        self,
        encoders: list,
        orches: list,
        agents: list,
        level_Ks: list[int],
        generative_head=None,
        meta_monitor=None,
    ):
        if generative_head is not None:
            raise NotImplementedError("L4 generative_head not yet implemented")
        if meta_monitor is not None:
            raise NotImplementedError("L5 meta_monitor not yet implemented")

        assert len(encoders) == len(orches) == len(agents), (
            "encoders, orches, and agents must have the same length"
        )
        self.encoders = encoders
        self.orches = orches
        self.agents = agents
        self.level_Ks = level_Ks
        self._step_ticks: list[int] = [0] * len(orches)
        self._epistemic: list[tuple[float, float] | None] = [None] * len(orches)

    def step(self, observation, l1_obs_dict: dict | None = None) -> dict:
        """Step all levels on one observation.

        L1 always fires. Level i (i>=1) fires when _step_ticks[0] % level_Ks[i] == 0
        (checked AFTER incrementing, so first fire is at step K).

        Args:
            observation: Domain-specific input passed through to each encoder.
            l1_obs_dict: Optional dict[agent_id, np.ndarray] overriding L1 routing.
                         Use for partitioned training (agent 0 gets obs_a, others get obs_b).

        Returns:
            Dict with keys "level1".."levelN". Non-firing levels return empty dict.
        """
        results: dict[str, dict] = {}
        n = len(self.orches)

        # Level 0 (L1): always fires
        if l1_obs_dict is not None:
            l1_result = self.orches[0].step(l1_obs_dict)
        else:
            vecs = self.encoders[0].encode(observation, epistemic=None)
            obs_dict = {a.agent_id: vecs[0] for a in self.agents[0]}
            l1_result = self.orches[0].step(obs_dict)
        self._step_ticks[0] += 1
        self._epistemic[0] = self._extract_epistemic(0, l1_result)
        results["level1"] = l1_result

        # Higher levels: cadence check on total step count
        for i in range(1, n):
            if self._step_ticks[0] % self.level_Ks[i] == 0:
                vecs = self.encoders[i].encode(observation, epistemic=self._epistemic[i - 1])
                last_result: dict = {}
                for vec in vecs:
                    obs_dict = {a.agent_id: vec for a in self.agents[i]}
                    last_result = self.orches[i].step(obs_dict)
                self._step_ticks[i] += 1
                self._epistemic[i] = self._extract_epistemic(i, last_result)
                results[f"level{i + 1}"] = last_result
            else:
                results[f"level{i + 1}"] = {}

        return results

    def _extract_epistemic(self, level_idx: int, step_result: dict) -> tuple[float, float]:
        """Extract (weight, epistemic_loss) from primary agent at this level.

        weight: mean pattern weight from store; 0.0 if store empty.
        epistemic_loss: from agent step result dict key 'epistemic_loss'; 0.0 if absent.
        """
        primary = self.agents[level_idx][0]
        records = primary.store.query(primary.agent_id)
        weight = float(np.mean([w for _, w in records])) if records else 0.0
        agent_result = step_result.get(primary.agent_id, {})
        epistemic_loss = float(agent_result.get("epistemic_loss", 0.0))
        return (weight, epistemic_loss)
