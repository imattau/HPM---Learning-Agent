from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from hpm_ai_v6.hpm_model.core.cell import Cell


class RelationRegistry:
    """
    Learns per-relation-type embedding vectors via TransE-style online updates.

    For relation r: update r += lr * (target_emb - source_emb - r)
    Converges to the mean (target - source) offset for that relation type.
    """

    def __init__(self, embedding_dim: int = 64, seed: int = 42):
        self.embedding_dim = embedding_dim
        self._rng = np.random.default_rng(seed)
        self._embeddings: Dict[str, np.ndarray] = {}

    def get_or_create(self, relation_name: str) -> np.ndarray:
        if relation_name not in self._embeddings:
            self._embeddings[relation_name] = self._rng.standard_normal(
                self.embedding_dim
            ).astype(np.float32)
        return self._embeddings[relation_name]

    def update(
        self,
        relation_name: str,
        source_emb: np.ndarray,
        target_emb: np.ndarray,
        lr: float = 0.01,
    ) -> None:
        src = np.asarray(source_emb, dtype=np.float32)
        tgt = np.asarray(target_emb, dtype=np.float32)
        if src.shape[0] != self.embedding_dim or tgt.shape[0] != self.embedding_dim:
            return
        r = self.get_or_create(relation_name)
        self._embeddings[relation_name] = r + lr * (tgt - src - r)

    def predict_target(self, source_emb: np.ndarray, relation_name: str) -> np.ndarray:
        src = np.asarray(source_emb, dtype=np.float32)
        r = self.get_or_create(relation_name)
        if src.shape[0] != self.embedding_dim:
            return src
        return src + r

    def similarity(self, rel_a: str, rel_b: str) -> float:
        a = self.get_or_create(rel_a)
        b = self.get_or_create(rel_b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def find_similar_relations(
        self, relation_name: str, top_k: int = 5
    ) -> List[Tuple[float, str]]:
        scored = [
            (self.similarity(relation_name, other), other)
            for other in self._embeddings
            if other != relation_name
        ]
        scored.sort(reverse=True)
        return scored[:top_k]

    def coherence_score(
        self,
        source_emb: np.ndarray,
        relation_name: str,
        target_emb: np.ndarray,
    ) -> float:
        """Cosine similarity between (source + relation) and target."""
        predicted = self.predict_target(source_emb, relation_name)
        tgt = np.asarray(target_emb, dtype=np.float32)
        np_pred, nt = np.linalg.norm(predicted), np.linalg.norm(tgt)
        if np_pred < 1e-9 or nt < 1e-9:
            return 0.0
        return float(np.dot(predicted, tgt) / (np_pred * nt))

    def populate_from_cells(self, cells: List[Cell]) -> None:
        """Load relation embeddings from dim-2 cells with 'rel_' prefix names."""
        for cell in cells:
            name = getattr(cell, "name", "")
            if name.startswith("rel_"):
                relation_name = name[4:]  # strip "rel_"
                self._embeddings[relation_name] = np.asarray(cell.as_numpy(), dtype=np.float32)

    def to_cells(self) -> List[Tuple[Cell, float]]:
        """Export relation embeddings as dim-2 cells."""
        return [
            (Cell(name=f"rel_{name}", dim=2, embedding=emb.tolist()), 1.0)
            for name, emb in self._embeddings.items()
        ]
