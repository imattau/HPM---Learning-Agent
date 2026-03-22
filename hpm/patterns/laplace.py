import uuid
import numpy as np


class LaplacePattern:
    """
    Pattern h = (mu, b) — a Laplace generative model over feature space.
    b is a per-dimension scale vector (all entries > 0, floored at 1e-6).
    Value type: update() returns a new instance; id is preserved on update,
    fresh on recombination.

    log_prob returns NLL (lower = more probable), consistent with GaussianPattern.
    No sigma attribute: routes sym_kl_normalised to Monte Carlo fallback via sample().
    """

    def __init__(self, mu: np.ndarray, b: np.ndarray, id: str | None = None,
                 level: int = 1, source_id: str | None = None, freeze_mu: bool = False):
        self.id = id or str(uuid.uuid4())
        self.mu = np.array(mu, dtype=float)
        self.b = np.maximum(np.array(b, dtype=float), 1e-6)
        self.level = level
        self._n_obs: int = 0
        self.source_id = source_id
        self.freeze_mu = freeze_mu

    def log_prob(self, x: np.ndarray) -> float:
        """Return NLL: sum(|x - mu| / b) + sum(log(2b)). Lower = more probable."""
        diff = np.asarray(x, dtype=float) - self.mu
        return float(np.sum(np.abs(diff) / self.b) + np.sum(np.log(2.0 * self.b)))

    def sample(self, n: int, rng) -> np.ndarray:
        """Return n samples from the Laplace distribution, shape (n, D)."""
        return rng.laplace(loc=self.mu, scale=self.b, size=(n, len(self.mu)))

    def update(self, x: np.ndarray) -> 'LaplacePattern':
        """Online update. Returns new instance (value-type semantics).

        mu: running mean (approximation to true median).
        b: running mean of |x - mu_old|, floored at 1e-6.
           Unlike GaussianPattern which keeps sigma fixed, b is updated
           because it is cheap to estimate online and converges to ~0 for
           degenerate distributions (all obs identical), which is correct.
        """
        mu_old = self.mu
        n = self._n_obs + 1
        new_mu = mu_old if self.freeze_mu else (mu_old * self._n_obs + np.asarray(x, dtype=float)) / n
        # Residual uses mu_old (before shift) to avoid downward bias on b
        new_b = np.maximum((self.b * self._n_obs + np.abs(np.asarray(x, dtype=float) - mu_old)) / n, 1e-6)
        new_p = LaplacePattern(new_mu, new_b, id=self.id, level=self.level,
                               source_id=self.source_id, freeze_mu=self.freeze_mu)
        new_p._n_obs = n
        return new_p

    def recombine(self, other: 'LaplacePattern') -> 'LaplacePattern':
        if not isinstance(other, LaplacePattern):
            raise TypeError(f"Cannot recombine LaplacePattern with {type(other).__name__}")
        return LaplacePattern(0.5 * self.mu + 0.5 * other.mu,
                              0.5 * self.b + 0.5 * other.b)

    def description_length(self) -> float:
        return float(np.sum(np.abs(self.mu) > 1e-6) + self.b.shape[0])

    def connectivity(self) -> float:
        """Always 0.0 — no off-diagonal structure in diagonal-scale parameterisation."""
        return 0.0

    def compress(self) -> float:
        """Concentration ratio: fraction of total scale in the dominant dimension.
        Analogous to the top-eigenvalue ratio in GaussianPattern.compress().
        Returns 1/D for uniform b (uncompressed), approaches 1.0 as scale concentrates.
        """
        total = self.b.sum()
        if total == 0.0:
            return 0.0
        return float(self.b.max() / total)

    def is_structurally_valid(self) -> bool:
        return bool(np.all(self.b > 0))

    def to_dict(self) -> dict:
        return {
            'type': 'laplace',
            'id': self.id,
            'mu': self.mu.tolist(),
            'b': self.b.tolist(),
            'n_obs': self._n_obs,
            'level': self.level,
            'source_id': self.source_id,
            'freeze_mu': self.freeze_mu,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'LaplacePattern':
        p = cls(np.array(d['mu']), np.array(d['b']), id=d['id'],
                level=d.get('level', 1), source_id=d.get('source_id', None),
                freeze_mu=d.get('freeze_mu', False))
        p._n_obs = d['n_obs']
        return p
