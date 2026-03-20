import uuid
import numpy as np
from scipy.stats import multivariate_normal


class GaussianPattern:
    """
    Pattern h = (mu, Sigma) — a Gaussian generative model over feature space.
    Value type: update() returns a new instance; id is preserved on update,
    fresh on recombination.
    """

    def __init__(self, mu: np.ndarray, sigma: np.ndarray, id: str | None = None, level: int = 1):
        self.id = id or str(uuid.uuid4())
        self.mu = np.array(mu, dtype=float)
        self.sigma = np.array(sigma, dtype=float)
        self.level = level
        self._n_obs: int = 0

    def log_prob(self, x: np.ndarray) -> float:
        return float(-multivariate_normal.logpdf(x, mean=self.mu, cov=self.sigma))

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
        eigenvalues = np.linalg.eigvalsh(self.sigma)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        total = eigenvalues.sum()
        if total == 0.0:
            return 0.0
        return float(eigenvalues[-1] / total)

    def update(self, x: np.ndarray) -> 'GaussianPattern':
        n = self._n_obs + 1
        new_mu = (self.mu * self._n_obs + x) / n
        new_p = GaussianPattern(new_mu, self.sigma.copy(), id=self.id, level=self.level)
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
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'GaussianPattern':
        p = cls(np.array(d['mu']), np.array(d['sigma']), id=d['id'],
                level=d.get('level', 1))
        p._n_obs = d['n_obs']
        return p
