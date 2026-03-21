import numpy as np
from .agent import Agent
from ..field.field import PatternField
from ..patterns.gaussian import GaussianPattern


class MultiAgentOrchestrator:
    """
    Coordinates multiple HPM agents through a shared PatternField.

    Agents step sequentially to avoid field update race conditions.
    Each agent sees field updates from agents that stepped before it.

    Shared seeding (for observational mode, spec B2):
    For cross-agent frequency signals to be non-trivial, agents must share some
    pattern UUIDs. Pass seed_pattern to re-seed all agents with a common pattern.
    GaussianPattern.update() preserves UUID, so the shared UUID persists across steps.

    M3 enforcement (spec M3):
    When only 1 agent is active, social evaluation is self-referential.
    The orchestrator detaches agent.field during step() so freq signals are 0.0,
    then re-registers patterns and restores field. 'm3_active' is set in metrics.

    Usage:
        field = PatternField()
        agents = [Agent(cfg_i, field=field) for cfg_i in configs]
        seed = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
        orch = MultiAgentOrchestrator(agents, field, seed_pattern=seed)
        history = orch.run(observation, n_steps=100)
    """

    def __init__(
        self,
        agents: list,
        field: PatternField,
        seed_pattern: GaussianPattern | None = None,
        groups: dict | None = None,   # agent_id -> group_id
        monitor=None,
        strategist=None,
    ):
        self.agents = agents
        self._groups = groups
        self._group_fields: dict[str, PatternField] = {}
        self._t = 0
        self.monitor = monitor
        self.strategist = strategist

        if groups is not None:
            # Create one PatternField per unique group and assign to agents
            for group_id in set(groups.values()):
                self._group_fields[group_id] = PatternField()
            for agent in agents:
                agent.field = self._group_fields[groups[agent.agent_id]]
            self.field = None   # ungrouped field unused when all agents are grouped
        else:
            self.field = field
            for agent in agents:
                if agent.field is None:
                    agent.field = field

        if seed_pattern is not None:
            self._seed_shared(seed_pattern)

    def group_field_quality(self) -> dict:
        """
        Returns field quality metrics per group, keyed by group_id.
        Delegates to PatternField.field_quality() for each group field.
        Returns {} when groups are not configured.
        """
        return {
            gid: gfield.field_quality()
            for gid, gfield in self._group_fields.items()
        }

    def _seed_shared(self, seed: GaussianPattern) -> None:
        """Re-seed all agents with a common GaussianPattern (same UUID).

        Each agent receives its own copy (same id, same parameters) to preserve
        copy semantics (spec B1) — no aliased objects across agent stores.
        """
        for agent in self.agents:
            # Remove existing patterns
            existing = agent.store.query(agent.agent_id)
            for p, _ in existing:
                agent.store.delete(p.id)
            # Save a per-agent copy with the same UUID — preserves copy semantics (B1)
            agent.store.save(
                GaussianPattern(seed.mu.copy(), seed.sigma.copy(), id=seed.id),
                1.0,
                agent.agent_id,
            )

    def step(
        self,
        observations: dict,
        rewards: dict | None = None,
    ) -> dict:
        """
        Step each agent sequentially. Returns per-agent metrics.

        M3 enforcement (spec M3): when only 1 agent is active, social evaluation
        is self-referential and must be gated off. We do this by temporarily
        detaching the agent's field reference during step() so freq signals return
        0.0, then re-registering the agent's patterns with the field afterwards.
        'm3_active' is set True in metrics to signal this condition.
        """
        if rewards is None:
            rewards = {}
        self._t += 1
        m3_active = len(self.agents) == 1
        metrics = {}
        for agent in self.agents:
            x = observations[agent.agent_id]
            r = rewards.get(agent.agent_id, 0.0)
            if m3_active:
                # Detach field so freq signals are 0 during step (spec M3)
                actual_field = agent.field
                agent.field = None
                try:
                    step_metrics = agent.step(x, reward=r)
                finally:
                    # Always restore field, even if step() raises (spec M3 — temporary detach only)
                    agent.field = actual_field
                # Re-register patterns with field for external observers
                if actual_field is not None:
                    records = agent.store.query(agent.agent_id)
                    actual_field.register(
                        agent.agent_id, [(p.id, w) for p, w in records]
                    )
                    # Run sharing check that was suppressed during the M3 detach.
                    # Overwrites communicated_out: 0 from agent.step() (field was None during step).
                    patterns_post = [p for p, _ in records]
                    step_metrics['communicated_out'] = agent._share_pending(actual_field, patterns_post)
            else:
                step_metrics = agent.step(x, reward=r)
            step_metrics["m3_active"] = m3_active
            metrics[agent.agent_id] = step_metrics

        # Communication phase -- within-group only when groups configured
        if self._group_fields:
            for group_id, gfield in self._group_fields.items():
                broadcasts = gfield.drain_broadcasts()
                group_agent_ids = {aid for aid, gid in self._groups.items() if gid == group_id}
                for source_agent_id, pattern in broadcasts:
                    for agent in self.agents:
                        if agent.agent_id in group_agent_ids and agent.agent_id != source_agent_id:
                            agent._accept_communicated(pattern, source_agent_id)
        elif self.field is not None:
            broadcasts = self.field.drain_broadcasts()
            for source_agent_id, pattern in broadcasts:
                for agent in self.agents:
                    if agent.agent_id != source_agent_id:
                        agent._accept_communicated(pattern, source_agent_id)

        # Aggregate total_conflict across all agents
        total_conflict_sum = sum(
            metrics[aid].get("total_conflict", 0.0) for aid in metrics
        )
        field_quality = (
            self.monitor.step(self._t, self.agents, total_conflict_sum)
            if self.monitor is not None
            else {}
        )

        interventions = (
            self.strategist.step(self._t, field_quality, self.agents)
            if self.strategist is not None
            else {}
        )

        return {**metrics, "field_quality": field_quality, "interventions": interventions}

    def run(
        self,
        observation: np.ndarray,
        n_steps: int,
        rewards: dict | None = None,
    ) -> list:
        """Run all agents on the same observation for n_steps."""
        obs = {a.agent_id: observation for a in self.agents}
        return [self.step(obs, rewards=rewards) for _ in range(n_steps)]
