import itertools
import os
from typing import Iterator

import numpy as np
import requests

from .base import hash_vectorise

_TOPIC_EXPRS: dict[str, list[str]] = {
    "algebra":       ["x**2 + x + 1", "x**2 - 4", "2*x + 3", "x**3 - x", "a*x**2 + b*x + c"],
    "calculus":      ["sin(x)", "exp(x)", "log(x)", "x**2 * sin(x)", "1/(1 + x**2)"],
    "statistics":    ["(x - mu)/sigma", "exp(-x**2/2)", "x*(1-x)", "n*p", "p**k * (1-p)**(n-k)"],
    "geometry":      ["pi*r**2", "4*pi*r**3/3", "sqrt(a**2 + b**2)", "2*pi*r", "b*h/2"],
    "number_theory": ["2**n - 1", "n*(n+1)/2", "gcd(a, b)", "a**p % p", "phi(n)"],
    "trigonometry":  ["sin(x)**2 + cos(x)**2", "sin(2*x)", "cos(x - y)", "tan(x)", "2*sin(x)*cos(x)"],
}


class MathSubstrate:
    """
    ExternalSubstrate backed by SymPy (required), SciPy constants (optional),
    and Wolfram Alpha (optional).

    Core dependency: sympy.
    Optional: scipy (physical constants), Wolfram Alpha API key.

    fetch(query): tries to parse as SymPy expression; if that fails, treats
    as topic name and returns vectors for known math topic expressions.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        timeout: float = 5.0,
        use_scipy: bool = True,
        wolfram_app_id: str | None = None,
    ):
        try:
            from sympy import sympify, SympifyError
            self._sympify = sympify
            self._SympifyError = SympifyError
        except ImportError:
            raise ImportError(
                "sympy is required for MathSubstrate. Install with: pip install sympy"
            )

        self.feature_dim = feature_dim
        self.timeout = timeout
        self._cache: dict[str, list[np.ndarray]] = {}

        self._scipy_constants = None
        if use_scipy:
            try:
                import scipy.constants
                self._scipy_constants = scipy.constants.physical_constants
            except ImportError:
                pass

        self._wolfram_app_id = wolfram_app_id or os.environ.get('WOLFRAM_APP_ID')

    def fetch(self, query: str) -> list[np.ndarray]:
        if query in self._cache:
            return self._cache[query]

        results = []

        # SymPy component: try to parse as expression first
        try:
            expr = self._sympify(query)
            results.append(hash_vectorise(str(expr), self.feature_dim))
        except self._SympifyError:
            # Treat as topic name
            normalised = query.lower().strip()
            topic_exprs = _TOPIC_EXPRS.get(normalised)
            if topic_exprs is None:
                # Partial match: first topic key that is a substring of the query
                for key, exprs in _TOPIC_EXPRS.items():
                    if key in normalised:
                        topic_exprs = exprs
                        break
            if topic_exprs is not None:
                for expr_str in topic_exprs:
                    results.append(hash_vectorise(expr_str, self.feature_dim))

        # SciPy constants component (optional)
        if self._scipy_constants is not None:
            q = query.lower()
            count = 0
            for name, (value, unit, uncertainty) in self._scipy_constants.items():
                if q in name.lower():
                    results.append(hash_vectorise(f"{name} {unit}", self.feature_dim))
                    count += 1
                    if count >= 5:
                        break

        # Wolfram Alpha component (optional)
        if self._wolfram_app_id:
            try:
                resp = requests.get(
                    'https://api.wolframalpha.com/v1/result',
                    params={'appid': self._wolfram_app_id, 'i': query},
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    results.append(hash_vectorise(resp.text, self.feature_dim))
            except Exception:
                pass

        self._cache[query] = results
        return results

    def field_frequency(self, pattern) -> float:
        query = str(getattr(pattern, 'label', None) or 'algebra')
        vecs = self.fetch(query)
        if not vecs:
            return 0.0
        dim = self.feature_dim
        mu = np.array(pattern.mu, dtype=float)
        if len(mu) > dim:
            mu = mu[:dim]
        elif len(mu) < dim:
            mu = np.pad(mu, (0, dim - len(mu)))
        mu_norm = np.linalg.norm(mu)
        if mu_norm == 0:
            return 0.0
        mu_unit = mu / mu_norm
        mean_sim = float(np.mean([np.dot(mu_unit, v) for v in vecs]))
        return float(np.clip((mean_sim + 1.0) / 2.0, 0.0, 1.0))

    def stream(self) -> Iterator[np.ndarray]:
        """Stream math expression vectors, interleaved with SciPy constant vectors if available."""
        topics = itertools.cycle(_TOPIC_EXPRS.keys())
        if self._scipy_constants is None:
            for topic in topics:
                # All _TOPIC_EXPRS keys are guaranteed to return non-empty results
                for v in self.fetch(topic):
                    yield v
        else:
            const_names = itertools.cycle(self._scipy_constants.keys())
            for topic in topics:
                vecs = self.fetch(topic)
                if not vecs:
                    continue
                for v in vecs:
                    yield v
                yield hash_vectorise(next(const_names), self.feature_dim)
