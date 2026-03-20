import json
import numpy as np
import urllib.request
from typing import Iterator

from .base import hash_vectorise


_VECTOR_DIM = 64  # matches WikipediaSubstrate vector dimension


class PyPISubstrate:
    """
    ExternalSubstrate backed by PyPI package metadata (spec §3.8).

    Fetches package metadata from the PyPI JSON API (no API key required).
    Package descriptions are vectorised with hash_vectorise for cross-process
    determinism. Optionally augmented by WikipediaSubstrate for richer context
    (augment_with_wikipedia flag stored but deferred — YAGNI).

    Implements the ExternalSubstrate protocol:
      fetch(query)             -> list[np.ndarray]
      field_frequency(pattern) -> float
      stream()                 -> Iterator[np.ndarray]

    Note:
        Agents using PyPISubstrate must be configured with feature_dim=64
        (= _VECTOR_DIM). This matches WikipediaSubstrate's vector dimension.
        If your agents use a different feature_dim, set _VECTOR_DIM in
        hpm/substrate/pypi.py to match your feature_dim.
        field_frequency() raises ValueError on dimension mismatch.

    Args:
        seed_packages: list of PyPI package names to fetch metadata for.
            E.g. ["numpy", "scipy", "torch", "scikit-learn"]
        cache: if True (default), cache fetched metadata to avoid repeated HTTP calls.
        augment_with_wikipedia: stored for future use — not yet implemented.
        similarity_threshold: cosine similarity threshold for field_frequency (default 0.5).
    """

    _PYPI_URL = "https://pypi.org/pypi/{name}/json"

    def __init__(
        self,
        seed_packages: list[str],
        cache: bool = True,
        augment_with_wikipedia: bool = False,
        similarity_threshold: float = 0.5,
    ):
        self.seed_packages = seed_packages
        self.cache = cache
        self.augment_with_wikipedia = augment_with_wikipedia
        self.similarity_threshold = similarity_threshold
        self._cache: dict[str, dict] = {}  # package_name -> metadata dict
        self._vectors: dict[str, np.ndarray] = {}  # package_name -> vector
        self._loaded = False

    def _load_all(self) -> None:
        """Fetch metadata for all seed packages (once, if caching)."""
        if self._loaded and self.cache:
            return
        for name in self.seed_packages:
            if name in self._cache and self.cache:
                continue
            try:
                url = self._PYPI_URL.format(name=name)
                with urllib.request.urlopen(url, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                self._cache[name] = data
            except Exception:
                # Package not found or network error — skip silently
                self._cache[name] = {}
        # Build vectors from cached metadata
        for name, data in self._cache.items():
            text = self._package_text(name, data)
            self._vectors[name] = hash_vectorise(text, dim=_VECTOR_DIM)
        self._loaded = True

    def _package_text(self, name: str, data: dict) -> str:
        """Combine package name + summary + keywords into text for vectorisation."""
        if not data:
            return name
        info = data.get("info", {})
        parts = [
            name,
            info.get("summary", ""),
            info.get("keywords", "") or "",
        ]
        return " ".join(p for p in parts if p)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def fetch(self, query: str) -> list[np.ndarray]:
        """
        Return vectorised descriptions of packages whose text matches query keywords.
        Matching is keyword-based (case-insensitive substring).
        """
        self._load_all()
        query_lower = query.lower()
        results = []
        for name, data in self._cache.items():
            text = self._package_text(name, data).lower()
            if any(word in text for word in query_lower.split()):
                vec = self._vectors.get(name)
                if vec is not None:
                    results.append(vec)
        return results

    def field_frequency(self, pattern) -> float:
        """
        Fraction of seed packages whose description vector has cosine similarity
        > similarity_threshold with pattern.mu.

        Raises ValueError if pattern.mu.shape[0] != _VECTOR_DIM (64).
        Agents must use feature_dim=64 when using PyPISubstrate.
        """
        self._load_all()
        if not self._vectors:
            return 0.0

        mu = np.array(pattern.mu, dtype=float)
        if mu.shape[0] != _VECTOR_DIM:
            raise ValueError(
                f"Pattern feature_dim={mu.shape[0]} does not match PyPISubstrate "
                f"vector_dim={_VECTOR_DIM}. Configure agents with feature_dim={_VECTOR_DIM} "
                f"when using PyPISubstrate, or set _VECTOR_DIM in hpm/substrate/pypi.py "
                f"to match your feature_dim."
            )

        count = sum(
            1 for vec in self._vectors.values()
            if self._cosine_similarity(mu, vec) > self.similarity_threshold
        )
        return count / len(self._vectors)

    def stream(self) -> Iterator[np.ndarray]:
        """Yield vectorised package descriptions one at a time."""
        self._load_all()
        yield from self._vectors.values()
