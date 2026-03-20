import numpy as np


def sym_kl_normalised(p, q, n_samples: int = 200, rng=None) -> float:
    """
    Symmetrised KL divergence between two GaussianPatterns, normalised to [0,1].
    kappa_{ij} in D5 — incompatibility measure.
    Uses Monte Carlo approximation for generality.

    Note: log_prob(x) returns -log p(x|h), so:
      log p(x|h) = -log_prob(x)
      KL(p||q) = E_p[log p - log q] = E_p[-p.log_prob(x) - (-q.log_prob(x))]
               = E_p[q.log_prob(x) - p.log_prob(x)]
    """
    if rng is None:
        rng = np.random.default_rng()

    samples_p = rng.multivariate_normal(p.mu, p.sigma, n_samples)
    kl_pq = float(np.mean([q.log_prob(s) - p.log_prob(s) for s in samples_p]))

    samples_q = rng.multivariate_normal(q.mu, q.sigma, n_samples)
    kl_qp = float(np.mean([p.log_prob(s) - q.log_prob(s) for s in samples_q]))

    sym_kl = max((kl_pq + kl_qp) / 2.0, 0.0)
    return float(sym_kl / (sym_kl + 1.0))   # normalise to [0, 1]


class MetaPatternRule:
    """
    D5: Discrete-time replicator dynamics with conflict inhibition.

    w_i(t+1) = w_i(t)
              + eta*(Total_i - Total_bar) * w_i(t)          # replicator
              - beta_c * sum_{j!=i} kappa_{ij} * w_i * w_j  # conflict inhibition

    Weights renormalised after each step.
    Floor: if all weights < epsilon, best pattern retained at 1.0.
    """

    def __init__(self, eta: float = 0.01, beta_c: float = 0.1, epsilon: float = 1e-4, kappa_D: float = 0.0):
        self.eta = eta
        self.beta_c = beta_c
        self.epsilon = epsilon
        self.kappa_D = kappa_D
        self._rng = np.random.default_rng(0)

    def step(self, patterns: list, weights: np.ndarray, totals: np.ndarray,
             densities=None, kappa_d_per_pattern=None) -> np.ndarray:
        n = len(patterns)
        if n == 0:
            return weights.copy()

        weights = np.array(weights, dtype=float)
        total_bar = float(np.dot(weights, totals))

        # Build incompatibility matrix kappa_{ij}
        kappa = np.zeros((n, n))  # diagonal stays 0: j!=i exclusion (D5)
        for i in range(n):
            for j in range(i + 1, n):  # explicit j != i
                k = sym_kl_normalised(patterns[i], patterns[j], rng=self._rng)
                kappa[i, j] = k
                kappa[j, i] = k
        assert np.all(np.diag(kappa) == 0.0), "kappa diagonal must be zero (j!=i in D5)"

        new_weights = weights.copy()
        for i in range(n):
            replicator = self.eta * (totals[i] - total_bar) * weights[i]
            conflict = self.beta_c * float(np.dot(kappa[i], weights) * weights[i])
            if densities is not None:
                kappa_d_i = kappa_d_per_pattern[i] if kappa_d_per_pattern is not None else self.kappa_D
                density_bias = kappa_d_i * densities[i] * weights[i]
            else:
                density_bias = 0.0
            new_weights[i] = weights[i] + replicator - conflict + density_bias

        new_weights = np.maximum(new_weights, 0.0)

        # Floor: never empty library (spec §3.3)
        if np.all(new_weights < self.epsilon):
            best = int(np.argmax(totals))
            new_weights = np.zeros(n)
            new_weights[best] = 1.0
            return new_weights

        total = new_weights.sum()
        if total > 0:
            new_weights /= total
        return new_weights
