import numpy as np
from hpm.patterns.gaussian import GaussianPattern


class CrossTaskRecombinator:
    """
    Off-line between-task recombinator.

    Pulls all Tier 2 patterns from TieredStore. For pairs whose cosine
    similarity falls in [similarity_lo, similarity_hi] — different enough
    to be complementary, similar enough to share structure — creates a
    midpoint meta-pattern and promotes it to Tier 2.

    This builds the structural intersection that HPM's hierarchical
    abstraction requires: e.g. "vertical symmetry" appearing in Task A
    and Task B recombines into a generalised Symmetry meta-pattern.
    """

    def __init__(self,
                 similarity_lo: float = 0.3,
                 similarity_hi: float = 0.9,
                 meta_weight: float = 0.3,
                 max_recombinants: int = 10):
        self.similarity_lo = similarity_lo
        self.similarity_hi = similarity_hi
        self.meta_weight = meta_weight
        self.max_recombinants = max_recombinants

    def consolidate(self, store, agent_id: str) -> int:
        """
        Run one consolidation pass on Tier 2 patterns for agent_id.
        Returns number of new meta-patterns promoted.
        """
        records = store.query_tier2(agent_id)
        if len(records) < 2:
            return 0

        promoted = 0
        existing_ids = {p.id for p, _, _ in store.query_tier2_all()}

        for i, (p1, w1) in enumerate(records):
            if promoted >= self.max_recombinants:
                break
            for j, (p2, w2) in enumerate(records):
                if j <= i:
                    continue
                sim = self._cosine_sim(p1.mu, p2.mu)
                if not (self.similarity_lo <= sim <= self.similarity_hi):
                    continue

                # Midpoint meta-pattern
                mu_meta = (p1.mu + p2.mu) / 2.0
                sigma_meta = (p1.sigma + p2.sigma) / 2.0
                p_meta = GaussianPattern(
                    mu=mu_meta,
                    sigma=sigma_meta,
                    level=max(getattr(p1, 'level', 1), getattr(p2, 'level', 1)) + 1,
                )

                if p_meta.id not in existing_ids:
                    store.promote_to_tier2(p_meta, self.meta_weight, agent_id)
                    existing_ids.add(p_meta.id)
                    promoted += 1

        return promoted

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
