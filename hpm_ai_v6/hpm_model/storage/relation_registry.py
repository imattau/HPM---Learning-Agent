from __future__ import annotations
import json
import os
from typing import Dict, List, Tuple
import numpy as np


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

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {k: v.tolist() for k, v in self._embeddings.items()}
        with open(path, "w") as f:
            json.dump({"embedding_dim": self.embedding_dim, "embeddings": data}, f)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.embedding_dim = int(data.get("embedding_dim", self.embedding_dim))
        self._embeddings = {
            k: np.array(v, dtype=np.float32)
            for k, v in data.get("embeddings", {}).items()
        }
