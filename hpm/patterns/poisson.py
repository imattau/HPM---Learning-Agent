import uuid
import numpy as np
from scipy.special import gammaln as _gammaln


class PoissonPattern:
    """
    Pattern h = (lam,) — a Poisson generative model over a D-dimensional
    non-negative integer space ℕ^D. Each dimension d is an independent
    Poisson(lambda_d) distribution.

    lam: shape (D,). lam[d] = lambda_d > 0 (mean of the Poisson).

    Update rule: online running mean (exact MLE for Poisson):
        lambda_new[d] = (lambda_old[d] * n_obs + x[d]) / (n_obs + 1)
    Initialise with lam = 1.0 (uninformative), _n_obs = 1.

    No sigma attribute: routes MetaPatternRule to Monte Carlo KL via sample().
    log_prob returns NLL (lower = more probable), consistent with GaussianPattern.
    """

    def __init__(self, lam: np.ndarray, id: str | None = None,
                 level: int = 1, source_id: str | None = None):
        self.id = id or str(uuid.uuid4())
        self.level = level
        self.source_id = source_id
        raw = np.asarray(lam, dtype=float)
        if raw.ndim != 1:
            raise ValueError(f"lam must be 1-D, got shape {raw.shape}")
        l = raw
        self.lam = np.maximum(l, 1e-8)
        # _n_obs = 1: one pseudo-observation at lambda=1 (uninformative prior)
        self._n_obs: int = 1

    @property
    def D(self) -> int:
        return self.lam.shape[0]

    def log_prob(self, x: np.ndarray) -> float:
        """Return NLL: -sum_d log Poisson(lambda_d).pmf(x_d).

        x: D-length non-negative integer array.
        NLL = sum_d [ lambda_d - x_d * log(lambda_d) + lgamma(x_d + 1) ]
        """
        x = np.asarray(x, dtype=float)
        nll = np.sum(self.lam - x * np.log(self.lam) + _gammaln(x + 1.0))
        return float(nll)

    def update(self, x: np.ndarray) -> 'PoissonPattern':
        """Online running-mean update. Returns new instance (value-type semantics).

        Uses pre-update lam (consistent with LaplacePattern convention).
        _n_obs incremented by 1.
        """
        x = np.asarray(x, dtype=float)
        n = self._n_obs
        lam_new = (self.lam * n + x) / (n + 1)
        lam_new = np.maximum(lam_new, 1e-8)
        new_p = PoissonPattern(lam_new, id=self.id,
                               level=self.level, source_id=self.source_id)
        new_p._n_obs = n + 1
        return new_p

    def recombine(self, other: 'PoissonPattern') -> 'PoissonPattern':
        """_n_obs-weighted average of lam vectors.

        Raises TypeError if other is not PoissonPattern.
        Raises ValueError if D mismatch.
        """
        if not isinstance(other, PoissonPattern):
            raise TypeError(f"Cannot recombine PoissonPattern with {type(other).__name__}")
        if self.D != other.D:
            raise ValueError(f"D mismatch: self.D={self.D}, other.D={other.D}")
        total = self._n_obs + other._n_obs
        if total == 0:
            lam_new = (self.lam + other.lam) / 2.0
        else:
            lam_new = (self.lam * self._n_obs + other.lam * other._n_obs) / total
        return PoissonPattern(np.maximum(lam_new, 1e-8))

    def sample(self, n: int, rng) -> np.ndarray:
        """Return shape (n, D) non-negative integer array."""
        result = np.empty((n, self.D), dtype=int)
        for d in range(self.D):
            result[:, d] = rng.poisson(self.lam[d], size=n)
        return result

    def description_length(self) -> float:
        """Count dimensions whose lambda deviates significantly from 1.0.

        lambda == 1 is the uninformative prior. Deviation indicates a learned rate.
        Threshold: |log(lambda_d)| > log(2) (i.e., lambda_d < 0.5 or > 2.0).
        """
        return float(np.sum(np.abs(np.log(self.lam)) > np.log(2.0)))

    def connectivity(self) -> float:
        """Returns 0.0 — independence assumption across dimensions."""
        return 0.0

    def compress(self) -> float:
        """max(lam) / mean(lam). Returns 1.0 if mean == 0."""
        mean_l = float(np.mean(self.lam))
        if mean_l == 0.0:
            return 1.0
        return float(np.max(self.lam)) / mean_l

    def is_structurally_valid(self) -> bool:
        """True iff all lam >= 1e-8."""
        return bool(np.all(self.lam >= 1e-8))

    def to_dict(self) -> dict:
        return {
            'type': 'poisson',
            'lam': self.lam.tolist(),
            'id': self.id,
            'level': self.level,
            'source_id': self.source_id,
            'n_obs': self._n_obs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'PoissonPattern':
        lam = np.array(d['lam'])
        p = cls(lam, id=d['id'],
                level=d.get('level', 1), source_id=d.get('source_id'))
        p._n_obs = d['n_obs']
        return p
