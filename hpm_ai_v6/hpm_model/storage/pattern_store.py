"""Shared cross-agent cross-corpus PatternStore.

Patterns are identified by embedding cosine similarity (threshold 0.85 by
default).  Weights are merged using a running average; observation counts are
tracked per entry.

Storage: <cache_dir>/shared/patterns.npz  (NumPy archive)

Array layout inside the .npz file
----------------------------------
vectors  : (N, D)  float32  -- embedding vectors
weights  : (N,)    float32  -- running-averaged weights
counts   : (N,)    int32    -- merge observation count
names    : (N,)    object   -- canonical pattern name (most recently seen)
agents   : (N,)    object   -- comma-separated agent names that contributed

Note on allow_pickle in np.load / np.savez
------------------------------------------
The `names` and `agents` arrays use numpy object dtype (Python strings).
NumPy requires allow_pickle=True to save/load these.  This is safe here
because we are the sole writer of the file -- no untrusted data is
deserialised.  The file lives under the project's own cache directory and
is never loaded from an external source.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from hpm_ai_v6.hpm_model.core.cell import Cell


class PatternStore:
    """Shared, cross-agent, cross-corpus pattern store."""

    def __init__(
        self,
        cache_dir: str,
        similarity_threshold: float = 0.85,
    ) -> None:
        """
        Args:
            cache_dir: Root cache directory (e.g. '.hpm_pattern_cache').
                       Store writes to <cache_dir>/shared/patterns.npz.
            similarity_threshold: Cosine similarity cutoff for merging.
                                  Patterns with sim >= threshold are treated
                                  as the same conceptual pattern.
        """
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        self._store_path = os.path.join(cache_dir, "shared", "patterns.npz")

        # In-memory arrays; None means not yet loaded/initialised.
        self._vectors: Optional[np.ndarray] = None   # (N, D) float32
        self._weights: Optional[np.ndarray] = None   # (N,) float32
        self._counts: Optional[np.ndarray] = None    # (N,) int32
        self._names: Optional[np.ndarray] = None     # (N,) object
        self._agents: Optional[np.ndarray] = None    # (N,) object

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_empty(self) -> bool:
        return self._vectors is None or len(self._vectors) == 0

    def _init_empty(self) -> None:
        """Reset in-memory state to a valid but empty store."""
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._weights = np.empty((0,), dtype=np.float32)
        self._counts = np.empty((0,), dtype=np.int32)
        self._names = np.empty((0,), dtype=object)
        self._agents = np.empty((0,), dtype=object)

    def _ensure_loaded(self) -> None:
        """Ensure in-memory state is initialised (load from disk if needed)."""
        if self._vectors is None:
            self.load()

    def _cosine_similarities(self, query: np.ndarray) -> np.ndarray:
        """Return cosine similarity of query against all stored vectors."""
        norms = np.linalg.norm(self._vectors, axis=1)  # (N,)
        query_norm = float(np.linalg.norm(query))
        return self._vectors @ query / (norms * query_norm + 1e-9)  # (N,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(
        self,
        patterns: List[Cell],
        weights: List[float],
        agent_name: str,
    ) -> None:
        """Merge a batch of (pattern, weight) pairs into the store.

        For each pair:
        - If the store is non-empty and the maximum cosine similarity against
          stored vectors is >= similarity_threshold, update the matching entry
          using a running average and increment its count.
        - Otherwise append a new entry.

        Does NOT call save() -- the caller must call save() after the batch.
        """
        self._ensure_loaded()

        for cell, w in zip(patterns, weights):
            vec = cell.as_numpy().astype(np.float32)
            D = vec.shape[0]

            if self._is_empty():
                self._vectors = vec.reshape(1, D)
                self._weights = np.array([w], dtype=np.float32)
                self._counts = np.array([1], dtype=np.int32)
                self._names = np.array([cell.name], dtype=object)
                self._agents = np.array([agent_name], dtype=object)
                continue

            stored_D = self._vectors.shape[1]
            if D != stored_D:
                raise ValueError(
                    f"Embedding dimension mismatch: incoming vector has dim {D}, "
                    f"store expects dim {stored_D}."
                )

            sims = self._cosine_similarities(vec)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim >= self.similarity_threshold:
                n = int(self._counts[best_idx])
                self._weights[best_idx] = (
                    float(self._weights[best_idx]) * n + w
                ) / (n + 1)
                self._counts[best_idx] = n + 1
                self._names[best_idx] = cell.name
                existing_agents = self._agents[best_idx].split(",")
                if agent_name not in existing_agents:
                    self._agents[best_idx] = ",".join(existing_agents + [agent_name])
            else:
                self._vectors = np.vstack([self._vectors, vec.reshape(1, D)])
                self._weights = np.append(self._weights, np.float32(w))
                self._counts = np.append(self._counts, np.int32(1))
                self._names = np.append(self._names, cell.name)
                self._agents = np.append(self._agents, agent_name)

    def load(self) -> Tuple[List[Cell], List[float]]:
        """Load all stored patterns and their averaged weights from disk.

        Returns ([], []) if no store file exists yet.
        Also populates the in-memory state so subsequent merge() calls work.
        """
        if not os.path.exists(self._store_path):
            self._init_empty()
            return [], []

        # allow_pickle=True is required for object-dtype string arrays.
        # Safe -- we are the sole writer; see module docstring.
        data = np.load(self._store_path, allow_pickle=True)
        self._vectors = data["vectors"].astype(np.float32)
        self._weights = data["weights"].astype(np.float32)
        self._counts = data["counts"].astype(np.int32)
        self._names = data["names"].astype(object)
        self._agents = data["agents"].astype(object)

        cells = [
            Cell(name=str(name), embedding=vec)
            for name, vec in zip(self._names, self._vectors)
        ]
        weights = [float(w) for w in self._weights]
        return cells, weights

    def find_similar(
        self,
        vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[float, Cell]]:
        """Return the top-k most similar stored patterns (descending similarity).

        Returns [] if the store is empty.
        """
        self._ensure_loaded()
        if self._is_empty():
            return []

        query = np.asarray(vector, dtype=np.float32)
        sims = self._cosine_similarities(query)
        k = min(top_k, len(sims))
        top_indices = np.argsort(sims)[::-1][:k]

        return [
            (float(sims[i]), Cell(name=str(self._names[i]), embedding=self._vectors[i]))
            for i in top_indices
        ]

    def save(self) -> None:
        """Persist in-memory state to <cache_dir>/shared/patterns.npz.

        Creates the directory if it does not exist. Overwrites any existing file.
        """
        self._ensure_loaded()
        os.makedirs(os.path.dirname(self._store_path), exist_ok=True)
        # allow_pickle=True required for object-dtype string arrays.
        # Safe -- we are the sole writer; see module docstring.
        np.savez(
            self._store_path,
            vectors=self._vectors,
            weights=self._weights,
            counts=self._counts,
            names=self._names,
            agents=self._agents,
        )

    def clear(self) -> None:
        """Reset in-memory store to empty (does not delete the file on disk)."""
        self._init_empty()
