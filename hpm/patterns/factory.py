import numpy as np
from hpm.patterns.gaussian import GaussianPattern
from hpm.patterns.laplace import LaplacePattern
from hpm.patterns.categorical import CategoricalPattern
from hpm.patterns.beta import BetaPattern
from hpm.patterns.poisson import PoissonPattern


def make_pattern(mu, scale, pattern_type: str = "gaussian", alphabet_size: int = 10, **kwargs):
    """Construct a pattern from (mu, scale) parameters.

    Args:
        mu: Location vector (ndarray or list). Used for dimensionality D = len(mu).
        scale: For Gaussian: covariance matrix. For Laplace: scale vector b
               (scalar is broadcast to np.ones(len(mu)) * scalar).
               Ignored for categorical, beta, and poisson patterns.
        pattern_type: "gaussian" | "laplace" | "categorical" | "beta" | "poisson".
        alphabet_size: K for CategoricalPattern; ignored by other types.
        **kwargs: Passed to the pattern constructor (id, level, source_id, freeze_mu).
                  For categorical, K can be passed explicitly via kwargs to override alphabet_size.
    """
    mu = np.asarray(mu, dtype=float)
    D = len(mu)
    if pattern_type == "gaussian":
        return GaussianPattern(mu, scale, **kwargs)
    elif pattern_type == "laplace":
        b = np.ones(D) * scale if np.isscalar(scale) else np.asarray(scale, dtype=float)
        return LaplacePattern(mu, b, **kwargs)
    elif pattern_type == "categorical":
        K = kwargs.pop("K", alphabet_size)
        probs = np.ones((D, K)) / K  # uniform = maximum entropy prior
        return CategoricalPattern(probs, K=K, **kwargs)
    elif pattern_type == "beta":
        # Uniform Beta(1,1) prior per dimension
        params = np.ones((D, 2))
        return BetaPattern(params, **kwargs)
    elif pattern_type == "poisson":
        # Uninformative lambda=1 prior per dimension
        lam = np.ones(D)
        return PoissonPattern(lam, **kwargs)
    else:
        raise ValueError(
            f"Unknown pattern_type: {pattern_type!r}. "
            "Expected 'gaussian', 'laplace', 'categorical', 'beta', or 'poisson'."
        )


def pattern_from_dict(d: dict):
    """Deserialise a pattern from a dict produced by to_dict().

    Dispatches on d['type']. Defaults to 'gaussian' if type key is absent
    (backward compatibility with pre-factory serialised patterns).
    """
    t = d.get('type', 'gaussian')
    if t == 'gaussian':
        return GaussianPattern.from_dict(d)
    elif t == 'laplace':
        return LaplacePattern.from_dict(d)
    elif t == 'categorical':
        return CategoricalPattern.from_dict(d)
    elif t == 'beta':
        return BetaPattern.from_dict(d)
    elif t == 'poisson':
        return PoissonPattern.from_dict(d)
    else:
        raise ValueError(
            f"Unknown pattern type in dict: {t!r}. "
            "Expected 'gaussian', 'laplace', 'categorical', 'beta', or 'poisson'."
        )
