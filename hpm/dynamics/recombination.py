import itertools
import numpy as np
from dataclasses import dataclass
from .meta_pattern_rule import sym_kl_normalised


@dataclass
class RecombinationResult:
    pattern: object
    insight_score: float
    parent_a_id: str
    parent_b_id: str
    trigger: str   # "time" | "conflict"


def _softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    x = scores / temp
    x = x - x.max()   # numerical stability
    e = np.exp(x)
    return e / e.sum()


class RecombinationOperator:
    """
    Stateless operator. attempt() selects Level 4+ parent pairs, creates h*
    via convex recombination, evaluates novelty + efficacy, and returns a
    RecombinationResult or None.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random generator. Accepts one for testability; defaults to a fresh
        unseeded generator if None.
    """

    def __init__(self, rng=None):
        self._rng = rng if rng is not None else np.random.default_rng()

    def attempt(self, patterns, weights, obs_buffer, config, trigger):
        """
        Parameters
        ----------
        patterns     : list of GaussianPattern (post-prune population)
        weights      : np.ndarray of corresponding weights
        obs_buffer   : list of np.ndarray (recent observations)
        config       : AgentConfig
        trigger      : str ("time" | "conflict")

        Returns
        -------
        RecombinationResult | None
        """
        # 1. Level gate
        candidate_indices = [i for i, p in enumerate(patterns) if p.level >= 4]
        if len(candidate_indices) < 2:
            return None

        # 2. Pair sampling
        weights = np.array(weights, dtype=float)
        pairs = list(itertools.combinations(candidate_indices, 2))
        pair_scores = np.array([weights[i] * weights[j] for i, j in pairs], dtype=float)
        pair_probs = _softmax(pair_scores, config.recomb_temp)

        for _ in range(config.N_recomb):
            idx = int(self._rng.choice(len(pairs), p=pair_probs))
            i, j = pairs[idx]
            parent_a = patterns[i]
            parent_b = patterns[j]

            kappa_ab = sym_kl_normalised(parent_a, parent_b, rng=self._rng)
            if kappa_ab >= config.kappa_max:
                continue   # incompatible pair — try another draw

            # 3. Crossover
            h_star = parent_a.recombine(parent_b)

            # 4. Feasibility gate (before insight computation)
            if not h_star.is_structurally_valid():
                continue

            # 5. Insight score
            nov = max(
                sym_kl_normalised(h_star, parent_a, rng=self._rng),
                sym_kl_normalised(h_star, parent_b, rng=self._rng),
            )
            if obs_buffer:
                # log_prob() returns NLL (positive); negating gives log-likelihood (≤ 0).
                # Eff is therefore non-positive, so insight ≤ beta_orig * alpha_nov * 1.0.
                # With defaults this caps entry_weight at kappa_0 * 0.5 = 0.05.
                eff = float(np.mean([-h_star.log_prob(x) for x in obs_buffer]))
            else:
                eff = 0.0

            insight = config.beta_orig * (config.alpha_nov * nov + config.alpha_eff * eff)

            # 6. Accept/discard — continue to next draw if this child is rejected
            if insight <= 0:
                continue

            return RecombinationResult(
                pattern=h_star,
                insight_score=insight,
                parent_a_id=parent_a.id,
                parent_b_id=parent_b.id,
                trigger=trigger,
            )

        return None   # all N_recomb draws rejected or discarded
