from typing import Iterator
import numpy as np
import requests
from .base import hash_vectorise

_WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


class WikipediaSubstrate:
    """
    ExternalSubstrate backed by the Wikipedia REST API.
    No API key required. Caches responses per query.
    Uses hash_vectorise to convert text summaries to fixed-dim float vectors.
    """

    def __init__(self, feature_dim: int = 32, timeout: float = 5.0):
        self.feature_dim = feature_dim
        self.timeout = timeout
        self._cache: dict[str, list[np.ndarray]] = {}

    def fetch(self, query: str) -> list[np.ndarray]:
        if query in self._cache:
            return self._cache[query]

        title = query.replace(' ', '_').title()
        url = _WIKI_API.format(title=title)
        try:
            resp = requests.get(url, timeout=self.timeout)
        except requests.RequestException:
            self._cache[query] = []
            return []

        if resp.status_code != 200:
            self._cache[query] = []
            return []

        extract = resp.json().get('extract', '')
        if not extract:
            self._cache[query] = []
            return []

        sentences = [s.strip() for s in extract.split('.') if s.strip()]
        vecs = [hash_vectorise(s, self.feature_dim) for s in sentences]
        self._cache[query] = vecs
        return vecs

    def field_frequency(self, pattern) -> float:
        """
        Mean cosine similarity between pattern.mu and vectorised Wikipedia summary.
        Uses pattern.label as query if present, else 'knowledge'.
        Returns float in [0, 1].
        """
        query = str(getattr(pattern, 'label', None) or 'knowledge')
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
        """Stream vectorised sentences from rotating Wikipedia articles."""
        topics = ["mathematics", "science", "history", "language", "cognition"]
        idx = 0
        while True:
            vecs = self.fetch(topics[idx % len(topics)])
            for v in vecs:
                yield v
            idx += 1
