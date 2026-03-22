import uuid
import numpy as np
from scipy.special import betaln as _betaln


class BetaPattern:
    """
    Pattern h = (params,) — a Beta generative model over a D-dimensional
    bounded-continuous space [0, 1]^D. Each dimension d is an independent
    Beta(alpha_d, beta_d) distribution.

    params: shape (D, 2). params[:, 0] = alpha, params[:, 1] = beta. Both > 0.

    Update rule: pseudo-count (conjugate-like) update treating x_d ∈ (0,1) as
    a fractional Bernoulli draw:
        alpha_new[d] = alpha_old[d] + x[d]
        beta_new[d]  = beta_old[d]  + (1 - x[d])
    This is the exact conjugate update for Bernoulli x_d ∈ {0,1}, and a natural
    online approximation for continuous x_d ∈ (0,1). Initialise with alpha=beta=1
    (uniform Beta) representing 2 pseudo-observations (_n_obs = 2).

    No sigma attribute: routes MetaPatternRule to Monte Carlo KL via sample().
    log_prob returns NLL (lower = more probable), consistent with GaussianPattern.
    """

    def __init__(self, params: np.ndarray, id: str | None = None,
                 level: int = 1, source_id: str | None = None):
        self.id = id or str(uuid.uuid4())
        self.level = level
        self.source_id = source_id
        p = np.array(params, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError(f"params must be shape (D, 2), got {p.shape}")
        self.params = np.maximum(p, 1e-8)
        # _n_obs = 2: one pseudo-count for alpha, one for beta (uniform prior)
        self._n_obs: int = 2

    @property
    def D(self) -> int:
        return self.params.shape[0]

    def log_prob(self, x: np.ndarray) -> float:
        """Return NLL: -sum_d log Beta(alpha_d, beta_d).pdf(x_d).

        x: D-length float array with values in (0, 1).
        Values are clipped to [1e-8, 1-1e-8] to prevent log(0).
        """
        x = np.clip(np.asarray(x, dtype=float), 1e-8, 1.0 - 1e-8)
        alpha = self.params[:, 0]
        beta  = self.params[:, 1]
        log_pdf = ((alpha - 1.0) * np.log(x)
                   + (beta - 1.0) * np.log(1.0 - x)
                   - _betaln(alpha, beta))
        return float(-np.sum(log_pdf))

    def update(self, x: np.ndarray) -> 'BetaPattern':
        """Pseudo-count update. Returns new instance (value-type semantics).

        alpha_new[d] += x[d]; beta_new[d] += (1 - x[d]).
        Uses pre-update params (consistent with LaplacePattern convention).
        """
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        params_new = self.params.copy()
        params_new[:, 0] += x
        params_new[:, 1] += (1.0 - x)
        params_new = np.maximum(params_new, 1e-8)
        new_p = BetaPattern(params_new, id=self.id,
                            level=self.level, source_id=self.source_id)
        new_p._n_obs = self._n_obs + 1
        return new_p

    def recombine(self, other: 'BetaPattern') -> 'BetaPattern':
        """_n_obs-weighted average of params matrices.

        Raises TypeError if other is not BetaPattern.
        Raises ValueError if D mismatch.
        """
        if not isinstance(other, BetaPattern):
            raise TypeError(f"Cannot recombine BetaPattern with {type(other).__name__}")
        if self.D != other.D:
            raise ValueError(f"D mismatch: self.D={self.D}, other.D={other.D}")
        total = self._n_obs + other._n_obs
        if total == 0:
            params_new = (self.params + other.params) / 2.0
        else:
            params_new = (self.params * self._n_obs + other.params * other._n_obs) / total
        return BetaPattern(np.maximum(params_new, 1e-8))

    def sample(self, n: int, rng) -> np.ndarray:
        """Return shape (n, D) float array in (0, 1)."""
        result = np.empty((n, self.D))
        for d in range(self.D):
            result[:, d] = rng.beta(self.params[d, 0], self.params[d, 1], size=n)
        return result

    def description_length(self) -> float:
        """Count dimensions with entropy below half the uniform Beta entropy.

        Uniform Beta(1,1) has maximum entropy log(B(1,1)) = 0. For Beta(a,b),
        entropy decreases as concentration (a+b) increases. We proxy
        'learned preference' by high concentration: alpha+beta > 4 (more than
        double the prior pseudo-count of 2).
        """
        concentration = self.params[:, 0] + self.params[:, 1]
        return float(np.sum(concentration > 4.0))

    def connectivity(self) -> float:
        """Returns 0.0 — independence assumption across dimensions."""
        return 0.0

    def compress(self) -> float:
        """max_concentration / mean_concentration.

        concentration[d] = alpha[d] + beta[d]. High ratio = one dimension
        much more sharply peaked than average. Returns 1.0 if mean == 0.
        """
        concentration = self.params[:, 0] + self.params[:, 1]
        mean_c = float(np.mean(concentration))
        if mean_c == 0.0:
            return 1.0
        return float(np.max(concentration)) / mean_c

    def is_structurally_valid(self) -> bool:
        """True iff all params >= 1e-8."""
        return bool(np.all(self.params >= 1e-8))

    def to_dict(self) -> dict:
        return {
            'type': 'beta',
            'params': self.params.tolist(),
            'id': self.id,
            'level': self.level,
            'source_id': self.source_id,
            'n_obs': self._n_obs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'BetaPattern':
        params = np.array(d['params'])
        p = cls(params, id=d['id'],
                level=d.get('level', 1), source_id=d.get('source_id'))
        p._n_obs = d['n_obs']
        return p
