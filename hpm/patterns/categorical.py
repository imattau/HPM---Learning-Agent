import uuid
import numpy as np


class CategoricalPattern:
    """
    Pattern h = (probs,) — a categorical generative model over a D-dimensional
    discrete space. Each dimension d is an independent categorical distribution
    over K symbols (alphabet size).

    Value type: update() returns a new instance; id is preserved on update,
    fresh on recombination.

    log_prob returns NLL (lower = more probable), consistent with GaussianPattern.
    No sigma attribute: routes sym_kl_normalised to Monte Carlo fallback via sample().
    """

    def __init__(self, probs: np.ndarray, K: int, id: str | None = None,
                 level: int = 1, source_id: str | None = None):
        if K < 2:
            raise ValueError(f"K must be >= 2, got {K}")
        self.id = id or str(uuid.uuid4())
        self.K = K
        self.level = level
        self.source_id = source_id
        # Floor at 1e-8, renormalise rows, then floor again to handle
        # edge case where renormalisation pushes a floored value below 1e-8
        p = np.maximum(np.array(probs, dtype=float), 1e-8)
        row_sums = p.sum(axis=1, keepdims=True)
        p = p / row_sums
        p = np.maximum(p, 1e-8)
        row_sums = p.sum(axis=1, keepdims=True)
        self.probs = p / row_sums
        # _n_obs initialised to K (Dirichlet pseudo-count — one per symbol per position)
        self._n_obs: int = K

    def log_prob(self, x: np.ndarray) -> float:
        """Return NLL: -sum(log(probs[d, x[d]])). Lower = more probable.

        x: D-length integer array with values in {0...K-1}.
        Out-of-range values raise IndexError (consistent with GaussianPattern
        not validating float inputs).
        """
        x = np.asarray(x, dtype=int)
        D = self.probs.shape[0]
        nll = 0.0
        for d in range(D):
            nll -= np.log(self.probs[d, x[d]])
        return float(nll)

    def update(self, x: np.ndarray) -> 'CategoricalPattern':
        """Online Bayesian count update. Returns new instance (value-type semantics).

        Uses pre-update probs (consistent with LaplacePattern's mu_old convention).
        _n_obs incremented by 1.
        """
        x = np.asarray(x, dtype=int)
        D, K = self.probs.shape
        n = self._n_obs
        probs_new = np.empty_like(self.probs)
        for d in range(D):
            # one-hot contribution for observed symbol
            one_hot = np.zeros(K)
            one_hot[x[d]] = 1.0
            probs_new[d] = (self.probs[d] * n + one_hot) / (n + 1)
        # Floor at 1e-8
        probs_new = np.maximum(probs_new, 1e-8)
        # Renormalise rows
        row_sums = probs_new.sum(axis=1, keepdims=True)
        probs_new = probs_new / row_sums
        new_p = CategoricalPattern(probs_new, K=self.K, id=self.id,
                                   level=self.level, source_id=self.source_id)
        new_p._n_obs = n + 1
        return new_p

    def recombine(self, other: 'CategoricalPattern') -> 'CategoricalPattern':
        """_n_obs-weighted average of probs matrices.

        Raises TypeError if other is not CategoricalPattern.
        Raises ValueError if D or K mismatch.
        """
        if not isinstance(other, CategoricalPattern):
            raise TypeError(f"Cannot recombine CategoricalPattern with {type(other).__name__}")
        if self.probs.shape != other.probs.shape:
            raise ValueError(
                f"Shape mismatch: self.probs.shape={self.probs.shape}, "
                f"other.probs.shape={other.probs.shape}"
            )
        if self.K != other.K:
            raise ValueError(f"K mismatch: self.K={self.K}, other.K={other.K}")

        total = self._n_obs + other._n_obs
        if total == 0:
            probs_new = (self.probs + other.probs) / 2.0
        else:
            probs_new = (self.probs * self._n_obs + other.probs * other._n_obs) / total

        # Renormalise rows (correct floating-point drift)
        row_sums = probs_new.sum(axis=1, keepdims=True)
        probs_new = probs_new / row_sums

        return CategoricalPattern(probs_new, K=self.K)

    def sample(self, n: int, rng) -> np.ndarray:
        """Return shape (n, D) integer array. Each column drawn independently."""
        D, K = self.probs.shape
        result = np.empty((n, D), dtype=int)
        for d in range(D):
            result[:, d] = rng.choice(K, size=n, p=self.probs[d])
        return result

    def description_length(self) -> float:
        """Count of positions whose entropy H(probs[d]) < log(K) * 0.5.

        Positions that have learned a definite preference (below half maximum entropy).
        """
        D, K = self.probs.shape
        log_K = np.log(K)
        count = 0
        for d in range(D):
            p = self.probs[d]
            # Entropy: H = -sum(p * log(p)), using only positive entries
            with np.errstate(divide='ignore', invalid='ignore'):
                h = -np.sum(p * np.log(np.where(p > 0, p, 1.0)))
            if h < log_K * 0.5:
                count += 1
        return float(count)

    def connectivity(self) -> float:
        """Returns 0.0 — independence assumption across positions."""
        return 0.0

    def compress(self) -> float:
        """max_row_entropy / mean_row_entropy.

        Returns 1.0 if mean_row_entropy == 0 (all rows are point masses).
        """
        D, K = self.probs.shape
        entropies = np.empty(D)
        for d in range(D):
            p = self.probs[d]
            with np.errstate(divide='ignore', invalid='ignore'):
                entropies[d] = -np.sum(p * np.log(np.where(p > 0, p, 1.0)))
        mean_h = float(np.mean(entropies))
        if mean_h == 0.0:
            return 1.0
        max_h = float(np.max(entropies))
        return max_h / mean_h

    def is_structurally_valid(self) -> bool:
        """True iff all probs >= 1e-8 and each row sums to 1 within 1e-6 tolerance."""
        if not np.all(self.probs >= 1e-8):
            return False
        row_sums = self.probs.sum(axis=1)
        return bool(np.all(np.abs(row_sums - 1.0) <= 1e-6))

    def to_dict(self) -> dict:
        return {
            'type': 'categorical',
            'probs': self.probs.tolist(),
            'K': self.K,
            'id': self.id,
            'level': self.level,
            'source_id': self.source_id,
            'n_obs': self._n_obs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CategoricalPattern':
        probs = np.array(d['probs'])
        K = d['K']
        p = cls(probs, K=K, id=d['id'],
                level=d.get('level', 1), source_id=d.get('source_id', None))
        p._n_obs = d['n_obs']
        return p
