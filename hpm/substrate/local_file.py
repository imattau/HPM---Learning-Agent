import os
from typing import Iterator
import numpy as np
from .base import hash_vectorise


class LocalFileSubstrate:
    """
    ExternalSubstrate backed by local .txt files.
    Deterministic — safe for offline testing and CI.
    Caches fetch results to avoid redundant file reads.
    """

    def __init__(self, directory: str, feature_dim: int = 32):
        self.directory = directory
        self.feature_dim = feature_dim
        self._cache: dict[str, list[np.ndarray]] = {}
        self._texts: list[str] = self._load_texts()

    def _load_texts(self) -> list[str]:
        texts = []
        for fname in sorted(os.listdir(self.directory)):
            fpath = os.path.join(self.directory, fname)
            if os.path.isfile(fpath) and fname.endswith('.txt'):
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
        return texts

    def fetch(self, query: str) -> list[np.ndarray]:
        if query in self._cache:
            return self._cache[query]
        q_lower = query.lower()
        results = [
            hash_vectorise(text, self.feature_dim)
            for text in self._texts
            if q_lower in text.lower()
        ]
        if not results:
            results = [hash_vectorise(t, self.feature_dim) for t in self._texts]
        self._cache[query] = results
        return results

    def field_frequency(self, pattern) -> float:
        """
        Fraction of documents with positive cosine similarity to pattern.mu.
        Returns float in [0, 1].
        """
        if not self._texts:
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
        count = sum(
            1 for text in self._texts
            if float(np.dot(mu_unit, hash_vectorise(text, dim))) > 0
        )
        return count / len(self._texts)

    def stream(self) -> Iterator[np.ndarray]:
        """Cycle through all text files indefinitely."""
        idx = 0
        while True:
            if not self._texts:
                return
            yield hash_vectorise(self._texts[idx % len(self._texts)], self.feature_dim)
            idx += 1
