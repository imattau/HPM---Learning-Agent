"""Fixed-dimensional Ollama embeddings for controlled HPM text experiments."""
from __future__ import annotations

import json
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np

from hpm_ai_v2.domains.text_domain import TextDomainConfig


class OllamaEmbeddingClient:
    """Small client for Ollama's batch embedding endpoint."""

    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:
        if not model.strip():
            raise ValueError("embedding model must not be empty")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed_many(self, texts: Iterable[str]) -> list[np.ndarray]:
        rows = list(texts)
        if not rows:
            return []
        request = Request(
            f"{self.base_url}/api/embed",
            data=json.dumps({"model": self.model, "input": rows}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Ollama embedding request failed for {self.model!r}: {exc}"
            ) from exc
        vectors = body.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(rows):
            raise ValueError("Ollama returned an unexpected number of embeddings")
        result = []
        for vector in vectors:
            arr = np.asarray(vector, dtype=np.float32)
            if arr.ndim != 1 or not np.all(np.isfinite(arr)):
                raise ValueError("Ollama returned an invalid embedding vector")
            norm = float(np.linalg.norm(arr))
            result.append(arr / norm if norm > 0 else arr)
        return result


class OllamaEmbeddingTextDomainConfig(TextDomainConfig):
    """TextDomainConfig whose concept slice is a frozen embedding vector.

    Vocabulary growth is disabled because adding token dimensions would change
    the embedding geometry. Callers should construct ReaderAgent with
    ``dynamic_vocab=False``.
    """

    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        s_dim: int = 20,
        dimension: int | None = None,
        cache: dict[str, list[float]] | None = None,
        client: OllamaEmbeddingClient | None = None,
    ) -> None:
        self.embedder = client or OllamaEmbeddingClient(
            model, base_url=base_url, timeout=timeout
        )
        self.embedding_model = self.embedder.model
        self.embedding_base_url = self.embedder.base_url
        self.embedding_timeout = self.embedder.timeout
        self._embedding_cache = {
            text: np.asarray(vector, dtype=np.float32)
            for text, vector in (cache or {}).items()
        }
        if dimension is None:
            probe = self.embedder.embed_many(["HPM embedding dimension probe"])[0]
            dimension = int(probe.size)
        if dimension < 1:
            raise ValueError("embedding dimension must be positive")
        if any(vector.size != dimension for vector in self._embedding_cache.values()):
            raise ValueError("cached embedding dimension does not match configuration")
        super().__init__(
            [f"EMBED_{index}" for index in range(dimension)],
            {},
            s_dim=s_dim,
        )

    def encode_passage(self, text: str) -> np.ndarray:
        vector = self._embedding_cache.get(text)
        if vector is None:
            vector = self.embedder.embed_many([text])[0]
            if vector.size != self.DIM:
                raise ValueError(
                    f"embedding dimension changed: expected {self.DIM}, received {vector.size}"
                )
            self._embedding_cache[text] = vector
        mu = np.zeros(self.m_dim, dtype=np.float32)
        mu[self.S_DIM : self.S_DIM + self.DIM] = vector
        return mu

    def precompute(self, texts: Iterable[str], batch_size: int = 32) -> None:
        """Batch-encode uncached text to reduce local model request overhead."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        unique = list(dict.fromkeys(text for text in texts if text not in self._embedding_cache))
        for start in range(0, len(unique), batch_size):
            batch = unique[start : start + batch_size]
            vectors = self.embedder.embed_many(batch)
            for text, vector in zip(batch, vectors):
                if vector.size != self.DIM:
                    raise ValueError(
                        f"embedding dimension changed: expected {self.DIM}, received {vector.size}"
                    )
                self._embedding_cache[text] = vector

    def add_word(self, word: str) -> int:
        raise RuntimeError("Ollama embedding dimensions are fixed; dynamic vocabulary is unsupported")

    def encoder_metadata(self, include_cache: bool = False) -> dict:
        metadata = {
            "type": "ollama_embedding",
            "model": self.embedding_model,
            "base_url": self.embedding_base_url,
            "timeout": self.embedding_timeout,
            "dimension": self.DIM,
        }
        if include_cache:
            metadata["cache"] = {
                text: vector.tolist() for text, vector in self._embedding_cache.items()
            }
        return metadata

    @classmethod
    def from_metadata(cls, metadata: dict) -> "OllamaEmbeddingTextDomainConfig":
        return cls(
            model=metadata["model"],
            base_url=metadata.get("base_url", "http://127.0.0.1:11434"),
            timeout=float(metadata.get("timeout", 120.0)),
            dimension=int(metadata["dimension"]),
            cache=metadata.get("cache", {}),
        )
