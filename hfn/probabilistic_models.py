# hfn/probabilistic_models.py
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


class ProbabilisticModel(ABC):
    """Abstract interface for probabilistic models inside an HFN node."""

    @abstractmethod
    def log_prob(self, x: np.ndarray) -> float:
        """Log probability of observation x under this model."""
        ...

    @abstractmethod
    def overlap(self, other: ProbabilisticModel) -> float:
        """Approximate overlap between this model and another."""
        ...

    @abstractmethod
    def update(self, x: np.ndarray, weight: float = 1.0, learning_rate: float = 0.1) -> None:
        """
        Update the model's parameters (mu, sigma, component means, etc.)
        based on observation x, weighted by how much this node contributed
        to explaining the observation.
        """
        ...

    @abstractmethod
    def get_state(self) -> dict:
        """Return a dictionary of parameters needed to reconstruct the model."""
        ...

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict) -> ProbabilisticModel:
        """Reconstruct a model from its state dictionary."""
        ...


class FlatGaussianModel(ProbabilisticModel):
    """Original single-diagonal-Gaussian model, extracted from HFN."""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray, use_diag: bool = True):
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.use_diag = use_diag
        self._sigma_diag = None
        self._log_det_cached = 0.0
        self._update_caches()

    def _update_caches(self):
        if self.use_diag:
            # sigma is already a D-vector diagonal — use directly
            self._sigma_diag = np.maximum(self.sigma, 1e-9)
            self._log_det_cached = float(np.sum(np.log(self._sigma_diag)))
        else:
            diag = np.diag(self.sigma)
            if np.allclose(self.sigma, np.diag(diag)):
                self._sigma_diag = np.maximum(diag, 1e-9)
                self._log_det_cached = float(np.sum(np.log(self._sigma_diag)))
            else:
                self._sigma_diag = None
                self._log_det_cached = 0.0

    def log_prob(self, x: np.ndarray) -> float:
        diff = np.asarray(x, dtype=float) - self.mu
        D = self.mu.shape[0]
        if self._sigma_diag is not None:
            # Diagonal case: -0.5 * (sum((x-mu)^2 / sigma) + sum(log(sigma)) + D*log(2pi))
            z2 = float(np.sum((diff * diff) / self._sigma_diag))
            return -0.5 * (z2 + self._log_det_cached + D * np.log(2.0 * np.pi))
        try:
            chol = np.linalg.cholesky(self.sigma)
            z = np.linalg.solve(chol, diff)
            log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
        except np.linalg.LinAlgError:
            diag = np.maximum(np.diag(self.sigma), 1e-9)
            z = diff / np.sqrt(diag)
            log_det = float(np.sum(np.log(diag)))
        return float(-0.5 * (np.dot(z, z) + log_det + D * np.log(2.0 * np.pi)))

    def overlap(self, other: ProbabilisticModel) -> float:
        if not isinstance(other, FlatGaussianModel):
            # fallback: compute using only means
            mu_other = getattr(other, 'mu', None)
            if mu_other is None: return 0.0
            diff = self.mu - mu_other
            return float(np.exp(-0.5 * np.dot(diff, diff)))
        diff = self.mu - other.mu
        if self._sigma_diag is not None and other._sigma_diag is not None:
            combined_diag = self._sigma_diag + other._sigma_diag
            # Use np.sum to ensure scalar result for diagonal Gaussian overlap
            val = np.sum((diff * diff) / combined_diag)
            return float(np.exp(-0.5 * float(val)))
        # Mixed case: expand diag to full matrix
        s_sigma = np.diag(self._sigma_diag) if (self.use_diag and self._sigma_diag is not None) else self.sigma
        o_sigma = np.diag(other._sigma_diag) if (other.use_diag and other._sigma_diag is not None) else other.sigma
        combined_sigma = s_sigma + o_sigma
        try:
            return float(np.exp(-0.5 * diff @ np.linalg.solve(combined_sigma, diff)))
        except np.linalg.LinAlgError:
            return 0.0

    def description_length(self) -> float:
        D = self.mu.shape[0]
        if self._sigma_diag is not None:
            return float(np.sum(np.abs(self.mu) > 1e-6)) + D
        return float(
            np.sum(np.abs(self.mu) > 1e-6)
            + np.sum(np.abs(self.sigma - np.diag(np.diag(self.sigma))) > 1e-6)
            + D
        )

    def update(self, x: np.ndarray, weight: float = 1.0, learning_rate: float = 0.1) -> None:
        """Online prototype learning via Exponential Moving Average (EMA)."""
        effective_lr = learning_rate * weight
        if effective_lr <= 0:
            return

        diff = x - self.mu
        # Update mu
        self.mu = (1.0 - effective_lr) * self.mu + effective_lr * x
        
        # Update sigma (running variance)
        if self.use_diag:
            self.sigma = (1.0 - effective_lr) * self.sigma + effective_lr * (diff * diff)
            self.sigma = np.maximum(self.sigma, 1e-9)
        else:
            self.sigma = (1.0 - effective_lr) * self.sigma + effective_lr * np.outer(diff, diff)
            # Ensure positive definiteness (simple nugget)
            np.fill_diagonal(self.sigma, np.diag(self.sigma) + 1e-9)
            
        self._update_caches()

    def get_state(self) -> dict:
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "use_diag": self.use_diag,
        }

    @classmethod
    def from_state(cls, state: dict) -> FlatGaussianModel:
        return cls(state["mu"], state["sigma"], state["use_diag"])


class GaussianMixtureModel(ProbabilisticModel):
    """
    Gaussian Mixture Model with K components.
    Uses Hebbian competitive learning for updates.
    """

    def __init__(self, components: list[FlatGaussianModel], weights: np.ndarray | list[float] | None = None):
        self.components = components
        if weights is None:
            self.weights = np.ones(len(components)) / len(components)
        else:
            self.weights = np.asarray(weights, dtype=float)
            self.weights /= self.weights.sum()

    @classmethod
    def from_params(cls, mus: list[np.ndarray], sigmas: list[np.ndarray], weights: list[float] | None = None, use_diag: bool = True):
        components = [FlatGaussianModel(mu, sigma, use_diag) for mu, sigma in zip(mus, sigmas)]
        return cls(components, weights)

    def log_prob(self, x: np.ndarray) -> float:
        log_weights = np.log(self.weights + 1e-12)
        log_probs = np.array([c.log_prob(x) for c in self.components])
        return float(np.logaddexp.reduce(log_weights + log_probs))

    def overlap(self, other: ProbabilisticModel) -> float:
        # Approximate overlap as max overlap of any component
        if isinstance(other, GaussianMixtureModel):
            return max(c1.overlap(c2) for c1 in self.components for c2 in other.components)
        return max(c.overlap(other) for c in self.components)

    def description_length(self) -> float:
        # Sum of components + weights - 1 (since weights sum to 1)
        return sum(c.description_length() for c in self.components) + len(self.weights) - 1

    def update(self, x: np.ndarray, weight: float = 1.0, learning_rate: float = 0.1) -> None:
        """Responsibility-weighted EMA updates for components and mixing weights."""
        effective_lr = learning_rate * weight
        if effective_lr <= 0:
            return

        log_weights = np.log(self.weights + 1e-12)
        log_probs = np.array([c.log_prob(x) for c in self.components])
        log_posterior = log_weights + log_probs
        log_posterior -= np.logaddexp.reduce(log_posterior)
        posterior = np.exp(log_posterior)

        # Update global mixing weights
        self.weights = (1.0 - effective_lr) * self.weights + effective_lr * posterior
        self.weights /= self.weights.sum()

        # Update components weighted by their responsibility
        for i, comp in enumerate(self.components):
            comp.update(x, weight=posterior[i] * weight, learning_rate=learning_rate)

    def get_state(self) -> dict:
        return {
            "weights": self.weights,
            "mus": [c.mu for c in self.components],
            "sigmas": [c.sigma for c in self.components],
            "use_diag": self.components[0].use_diag,
        }

    @classmethod
    def from_state(cls, state: dict) -> GaussianMixtureModel:
        components = []
        for mu, sigma in zip(state["mus"], state["sigmas"]):
            components.append(FlatGaussianModel(mu, sigma, state["use_diag"]))
        return cls(components, weights=state["weights"])


# --- Model Registry ---

_MODEL_REGISTRY: dict[str, type[ProbabilisticModel]] = {}


def register_model(name: str, cls: type[ProbabilisticModel]):
    _MODEL_REGISTRY[name] = cls


def get_model_class(name: str) -> type[ProbabilisticModel]:
    return _MODEL_REGISTRY.get(name, FlatGaussianModel)


def get_model_name(cls: type[ProbabilisticModel]) -> str:
    for name, c in _MODEL_REGISTRY.items():
        if c == cls:
            return name
    return cls.__name__


register_model("flat_gaussian", FlatGaussianModel)
register_model("gaussian_mixture", GaussianMixtureModel)
