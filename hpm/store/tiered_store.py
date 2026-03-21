from hpm.store.memory import InMemoryStore


class TieredStore:
    """
    Two-tier context-aware PatternStore.

    Tier 1 (ephemeral): per-task patterns, mutated freely by agent signal.
    Tier 2 (persistent): meta-patterns, protected from task signal mutation.

    Weight updates (update_weight) only affect Tier 1. Tier 2 is updated
    exclusively via similarity_merge() and promote_to_tier2().
    """

    def __init__(self):
        self._tier1: dict[str, InMemoryStore] = {}
        self._tier2: InMemoryStore = InMemoryStore()
        self._current_context: str | None = None

    def begin_context(self, context_id: str) -> None:
        """Start a new task context. Creates a fresh Tier 1 store."""
        self._current_context = context_id
        self._tier1[context_id] = InMemoryStore()

    def end_context(self, context_id: str, correct: bool) -> None:
        """End task context. Runs similarity_merge on success before clearing."""
        if correct and context_id in self._tier1:
            self.similarity_merge(context_id)
        self._tier1.pop(context_id, None)
        if self._current_context == context_id:
            self._current_context = None

    def save(self, pattern, weight: float, agent_id: str) -> None:
        if self._current_context is not None:
            self._tier1[self._current_context].save(pattern, weight, agent_id)
        else:
            self._tier2.save(pattern, weight, agent_id)

    def load(self, pattern_id: str) -> tuple:
        if self._current_context and pattern_id in self._tier1[self._current_context]._data:
            return self._tier1[self._current_context].load(pattern_id)
        return self._tier2.load(pattern_id)

    def query(self, agent_id: str) -> list:
        """Returns Tier 1 (current context) + Tier 2 patterns for agent."""
        t1 = []
        if self._current_context and self._current_context in self._tier1:
            t1 = self._tier1[self._current_context].query(agent_id)
        t2 = self._tier2.query(agent_id)
        return t1 + t2

    def query_all(self) -> list:
        t1 = []
        if self._current_context and self._current_context in self._tier1:
            t1 = self._tier1[self._current_context].query_all()
        return t1 + self._tier2.query_all()

    def query_tier2(self, agent_id: str) -> list:
        """Direct access to Tier 2 patterns (for monitoring/testing)."""
        return self._tier2.query(agent_id)

    def query_tier2_all(self) -> list:
        return self._tier2.query_all()

    def delete(self, pattern_id: str) -> None:
        if self._current_context and self._current_context in self._tier1:
            self._tier1[self._current_context].delete(pattern_id)

    def update_weight(self, pattern_id: str, weight: float) -> None:
        """Only mutates Tier 1. Tier 2 is protected from task signal."""
        if self._current_context and self._current_context in self._tier1:
            t1 = self._tier1[self._current_context]
            if pattern_id in t1._data:
                t1.update_weight(pattern_id, weight)
                return
        # Pattern is in Tier 2 — do not mutate (protection invariant)

    def similarity_merge(self, context_id: str,
                         similarity_threshold: float = 0.95,
                         consolidation_boost: float = 0.1,
                         max_tier2_patterns: int = 200) -> None:
        """
        Compare Tier 1 patterns against Tier 2.
        - If similar Tier 2 pattern found: boost its weight.
        - If no match and Tier 2 not full: promote pattern to Tier 2.
        """
        import numpy as np

        if context_id not in self._tier1:
            return

        t1_records = self._tier1[context_id].query_all()
        t2_all = self._tier2.query_all()
        t2_patterns = list(t2_all)

        for p1, w1, aid1 in t1_records:
            mu1 = p1.mu
            norm1 = np.linalg.norm(mu1)
            if norm1 < 1e-8:
                continue

            best_sim = -1.0
            best_t2_id = None
            best_t2_w = 0.0

            for p2, w2, _ in t2_patterns:
                mu2 = p2.mu
                norm2 = np.linalg.norm(mu2)
                if norm2 < 1e-8:
                    continue
                sim = float(np.dot(mu1, mu2) / (norm1 * norm2))
                if sim > best_sim:
                    best_sim = sim
                    best_t2_id = p2.id
                    best_t2_w = w2

            if best_sim >= similarity_threshold and best_t2_id is not None:
                self._tier2.update_weight(best_t2_id, best_t2_w + consolidation_boost)
            elif len(t2_patterns) < max_tier2_patterns:
                self._tier2.save(p1, w1 * 0.5, aid1)
                t2_patterns.append((p1, w1 * 0.5, aid1))

    def promote_to_tier2(self, pattern, weight: float, agent_id: str) -> None:
        """Directly promote a pattern to Tier 2."""
        self._tier2.save(pattern, weight, agent_id)
