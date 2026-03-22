import uuid
import numpy as np


class GaussianPattern:
    """
    Pattern h = (mu, Sigma) — a Gaussian generative model over feature space.
    Value type: update() returns a new instance; id is preserved on update,
    fresh on recombination.
    """

    def __init__(self, mu: np.ndarray, sigma: np.ndarray, id: str | None = None, level: int = 1, source_id: str | None = None, freeze_mu: bool = False):
        self.id = id or str(uuid.uuid4())
        self.mu = np.array(mu, dtype=float)
        self.sigma = np.array(sigma, dtype=float)
        self.level = level
        self._n_obs: int = 0
        self.source_id = source_id
        self.freeze_mu = freeze_mu
        self._chol: np.ndarray | None = None
        self._log_det: float | None = None
        self._eigenvalues: np.ndarray | None = None

    def _ensure_chol(self) -> None:
        """Cache Cholesky decomposition of sigma (computed once per pattern object)."""
        if self._chol is None:
            try:
                self._chol = np.linalg.cholesky(self.sigma)
                self._log_det = 2.0 * float(np.sum(np.log(np.diag(self._chol))))
            except np.linalg.LinAlgError:
                # Fallback: use diagonal approximation
                diag = np.maximum(np.diag(self.sigma), 1e-9)
                self._chol = np.diag(np.sqrt(diag))
                self._log_det = float(np.sum(np.log(diag)))

    def log_prob(self, x: np.ndarray) -> float:
        """Return NLL: -log p(x | mu, sigma). Lower = more probable."""
        self._ensure_chol()
        diff = np.asarray(x, dtype=float) - self.mu
        z = np.linalg.solve(self._chol, diff)
        D = self.mu.shape[0]
        return float(0.5 * (np.dot(z, z) + self._log_det + D * np.log(2.0 * np.pi)))

    def description_length(self) -> float:
        return float(
            np.sum(np.abs(self.mu) > 1e-6)
            + np.sum(np.abs(self.sigma - np.diag(np.diag(self.sigma))) > 1e-6)
            + self.sigma.shape[0]
        )

    def connectivity(self) -> float:
        n = self.sigma.shape[0]
        if n <= 1:
            return 0.0
        std = np.sqrt(np.diag(self.sigma))
        with np.errstate(invalid='ignore'):
            corr = self.sigma / np.outer(std, std)
        corr = np.nan_to_num(corr)
        mask = ~np.eye(n, dtype=bool)
        return float(np.mean(np.abs(corr[mask])))

    def compress(self) -> float:
        if self._eigenvalues is None:
            self._eigenvalues = np.linalg.eigvalsh(self.sigma)
        eigenvalues = np.maximum(self._eigenvalues, 0.0)
        total = eigenvalues.sum()
        if total == 0.0:
            return 0.0
        return float(eigenvalues[-1] / total)

    def update(self, x: np.ndarray) -> 'GaussianPattern':
        n = self._n_obs + 1
        new_mu = self.mu if self.freeze_mu else (self.mu * self._n_obs + x) / n
        new_p = GaussianPattern(new_mu, self.sigma.copy(), id=self.id, level=self.level, source_id=self.source_id, freeze_mu=self.freeze_mu)
        new_p._n_obs = n
        return new_p

    def recombine(self, other: 'GaussianPattern') -> 'GaussianPattern':
        alpha = 0.5
        new_mu = alpha * self.mu + (1 - alpha) * other.mu
        new_sigma = alpha * self.sigma + (1 - alpha) * other.sigma
        return GaussianPattern(new_mu, new_sigma)

    def is_structurally_valid(self) -> bool:
        try:
            eigenvalues = np.linalg.eigvalsh(self.sigma)
            return bool(np.all(eigenvalues > 0))
        except np.linalg.LinAlgError:
            return False

    def to_dict(self) -> dict:
        return {
            'type': 'gaussian',
            'id': self.id,
            'mu': self.mu.tolist(),
            'sigma': self.sigma.tolist(),
            'n_obs': self._n_obs,
            'level': self.level,
            'source_id': self.source_id,
            'freeze_mu': self.freeze_mu,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'GaussianPattern':
        p = cls(np.array(d['mu']), np.array(d['sigma']), id=d['id'],
                level=d.get('level', 1), source_id=d.get('source_id', None),
                freeze_mu=d.get('freeze_mu', False))
        p._n_obs = d['n_obs']
        return p
