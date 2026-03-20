import math


class PatternField:
    """
    Shared pattern field tracking pattern population across agents (spec §5.1, D6).

    Each agent registers its current (pattern_id, weight) pairs after each step.
    The field computes normalised frequency for each pattern UUID:

        freq_i(t) = weight_sum_for_uuid_i / total_weight_mass

    Pattern objects are never shared — only UUIDs and weights are broadcast.
    Implements the observational interaction mode from spec §5.1.

    For cross-agent freq signals to be non-trivial, agents must share some pattern UUIDs.
    Since GaussianPattern.update() preserves UUID (id=self.id), agents seeded with the
    same initial pattern will maintain shared UUID tracking across all steps.

    Field quality metrics (spec §5.2):
    - diversity: Shannon entropy of normalised pattern weight distribution
    - redundancy: 0.0 placeholder (pairwise KL deferred to Phase 4)
    """

    def __init__(self):
        # Maps agent_id -> {pattern_id: weight}
        self._agent_patterns: dict[str, dict[str, float]] = {}

    @property
    def n_agents(self) -> int:
        return len(self._agent_patterns)

    def register(self, agent_id: str, patterns_weights: list[tuple[str, float]]) -> None:
        """
        Update the field with an agent's current pattern UUIDs and weights.
        Replaces any previous registration for this agent.
        """
        self._agent_patterns[agent_id] = {pid: w for pid, w in patterns_weights}

    def _total_mass(self) -> float:
        return sum(
            w
            for agent_patterns in self._agent_patterns.values()
            for w in agent_patterns.values()
        )

    def freq(self, pattern_id: str) -> float:
        """
        Normalised frequency of pattern_id across the agent population.
        Returns 0.0 if pattern_id unknown or field empty.
        """
        mass = self._total_mass()
        if mass <= 0.0:
            return 0.0
        weight_sum = sum(
            agent_patterns.get(pattern_id, 0.0)
            for agent_patterns in self._agent_patterns.values()
        )
        return weight_sum / mass

    def freqs_for(self, pattern_ids: list[str]) -> list[float]:
        """Return normalised frequency for each pattern_id in the list."""
        mass = self._total_mass()
        if mass <= 0.0:
            return [0.0] * len(pattern_ids)
        result = []
        for pid in pattern_ids:
            weight_sum = sum(
                agent_patterns.get(pid, 0.0)
                for agent_patterns in self._agent_patterns.values()
            )
            result.append(weight_sum / mass)
        return result

    def field_quality(self) -> dict:
        """
        Field quality metrics (spec §5.2).

        diversity: Shannon entropy of normalised pattern weight distribution.
                   High diversity = agents maintain different patterns.
        redundancy: 0.0 placeholder (pairwise KL deferred to Phase 4).
        """
        mass = self._total_mass()
        if mass <= 0.0:
            return {"diversity": 0.0, "redundancy": 0.0}

        # Aggregate all pattern weights across all agents
        all_weights: dict[str, float] = {}
        for agent_patterns in self._agent_patterns.values():
            for pid, w in agent_patterns.items():
                all_weights[pid] = all_weights.get(pid, 0.0) + w

        # Shannon entropy over normalised distribution
        entropy = 0.0
        for w in all_weights.values():
            p = w / mass
            if p > 0.0:
                entropy -= p * math.log(p)

        return {"diversity": entropy, "redundancy": 0.0}
