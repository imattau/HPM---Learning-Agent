"""Text domain: TF-IDF encoded passages as HFN pattern substrate."""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import List, Dict
import numpy as np
from hpm_ai_v2.domains.base import DomainConfig


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "it", "its", "this", "that",
    "i", "we", "you", "he", "she", "they", "my", "our", "your",
}


def tokenise(text: str) -> List[str]:
    return [
        w for w in re.findall(r"[a-z]+", text.lower())
        if w not in STOPWORDS and len(w) > 2
    ]


class TextDomainConfig(DomainConfig):
    def __init__(self, concepts: List[str], idf: Dict[str, float], s_dim: int = 20):
        super().__init__(concepts, s_dim=s_dim)
        self.idf = idf
        self._passages: List[str] = []
        self._passage_vecs: List[np.ndarray] = []

    @classmethod
    def from_passages(cls, passages: List[str], max_vocab: int = 200, s_dim: int = 20) -> "TextDomainConfig":
        doc_freq: Counter = Counter()
        for p in passages:
            doc_freq.update(set(tokenise(p)))
        vocab = [w for w, _ in doc_freq.most_common(max_vocab)]
        n_docs = max(len(passages), 1)
        idf = {w: math.log((n_docs + 1) / (doc_freq[w] + 1)) + 1.0 for w in vocab}
        return cls(vocab, idf, s_dim=s_dim)

    def encode_passage(self, text: str) -> np.ndarray:
        tokens = tokenise(text)
        tf: Counter = Counter(tokens)
        n = max(len(tokens), 1)
        concept_vec = np.zeros(self.DIM)
        for i, word in enumerate(self.concepts):
            if word in tf:
                concept_vec[i] = (tf[word] / n) * self.idf.get(word, 1.0)
        norm = np.linalg.norm(concept_vec)
        if norm > 0:
            concept_vec /= norm
        mu = np.zeros(self.m_dim)
        mu[self.S_DIM: self.S_DIM + self.DIM] = concept_vec
        return mu

    def register_passage(self, text: str) -> int:
        idx = len(self._passages)
        self._passages.append(text)
        self._passage_vecs.append(self.encode_passage(text))
        return idx

    def get_passage(self, idx: int) -> str:
        return self._passages[idx]
