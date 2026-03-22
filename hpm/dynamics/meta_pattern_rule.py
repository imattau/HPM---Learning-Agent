import numpy as np
from collections import namedtuple

StepResult = namedtuple('StepResult', ['weights', 'total_conflict'])


def _gaussian_kl(p, q) -> float:
    """Closed-form KL(p||q) for multivariate Gaussians using cached Cholesky."""
    p._ensure_chol()
    q._ensure_chol()
    d = len(p.mu)
    diff = q.mu - p.mu
    # trace(q.sigma^{-1} @ p.sigma) = ||L_q^{-1} @ L_p||_F^2
    L_q_inv_L_p = np.linalg.solve(q._chol, p._chol)
    trace_term = float(np.sum(L_q_inv_L_p ** 2))
    # maha = diff^T q.sigma^{-1} diff = ||L_q^{-1} diff||^2
    z = np.linalg.solve(q._chol, diff)
    maha = float(np.dot(z, z))
    return max(0.0, 0.5 * (trace_term + maha - d + float(q._log_det - p._log_det)))


def sym_kl_normalised(p, q, n_samples: int = 200, rng=None) -> float:
    """
    Symmetrised KL divergence between two patterns, normalised to [0,1].
    kappa_{ij} in D5 — incompatibility measure.

    Uses closed-form solution for GaussianPatterns (exact, numerically stable).
    Falls back to Monte Carlo for non-Gaussian patterns.
    """
    if hasattr(p, 'mu') and hasattr(p, 'sigma') and hasattr(q, 'mu') and hasattr(q, 'sigma'):
        kl_pq = _gaussian_kl(p, q)
        kl_qp = _gaussian_kl(q, p)
    else:
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
             densities=None, kappa_d_per_pattern=None) -> StepResult:
        n = len(patterns)
        if n == 0:
            return StepResult(weights.copy(), 0.0)

        weights = np.array(weights, dtype=float)
        total_bar = float(np.dot(weights, totals))

        # Build incompatibility matrix kappa_{ij}
        # Skip KL computation for large stores (n > 20) to keep cost bounded.
        # Weight dynamics still run; only the conflict penalty is suppressed.
        kappa = np.zeros((n, n))  # diagonal stays 0: j!=i exclusion (D5)
        if n <= 20:
            for i in range(n):
                for j in range(i + 1, n):  # explicit j != i
                    k = sym_kl_normalised(patterns[i], patterns[j], rng=self._rng)
                    kappa[i, j] = k
                    kappa[j, i] = k
        assert np.all(np.diag(kappa) == 0.0), "kappa diagonal must be zero (j!=i in D5)"

        total_conflict = float(self.beta_c * float(weights @ kappa @ weights))

        if kappa_d_per_pattern is not None:
            assert len(kappa_d_per_pattern) == n, (
                f"kappa_d_per_pattern length {len(kappa_d_per_pattern)} != n_patterns {n}"
            )

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
            return StepResult(new_weights, 0.0)

        total = new_weights.sum()
        if total > 0:
            new_weights /= total
        return StepResult(new_weights, total_conflict)
