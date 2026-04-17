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
    # Allow 1-letter words for tests, but filter stopwords
    return [
        w for w in re.findall(r"[a-z]+", text.lower())
        if w not in STOPWORDS
    ]


def tokenise_raw(text: str) -> List[str]:
    """Tokenise including stopwords and punctuation (for syntactic analysis)."""
    # Simple whitespace splitting for now
    return [w.strip() for w in re.findall(r"\w+|[^\w\s]", text.lower()) if w.strip()]


class TextDomainConfig(DomainConfig):
    def __init__(self, concepts: List[str], idf: Dict[str, float], s_dim: int = 20, include_char_primitives: bool = False):
        if include_char_primitives:
            # Add character primitives
            for c in "abcdefghijklmnopqrstuvwxyz": concepts.append(f"CHAR_{c}")
            for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ": concepts.append(f"CHAR_{c}")
            for d in range(10): concepts.append(f"CHAR_DIGIT_{d}")
            # Add string utility primitives
            concepts.extend([
                "TO_UPPER", "TO_LOWER", "CHAR_EQ", "STRING_LEN", 
                "CHAR_AT", "EDIT_DISTANCE", "FIND_CLOSEST"
            ])
            
        super().__init__(concepts, s_dim=s_dim)
        self.idf = idf
        self._passages: List[str] = []
        self._passage_vecs: List[np.ndarray] = []
        self.include_char_primitives = include_char_primitives

    def get_pos_primitives(self) -> List[str]:
        return [
            "POS_NOUN", "POS_VERB", "POS_ADVERB", "POS_ADJECTIVE",
            "POS_DET", "POS_PREP", "POS_CONJ", "POS_PUNCT"
        ]

    def get_srl_primitives(self) -> List[str]:
        return [
            "SRL_AGENT", "SRL_PATIENT", "SRL_INSTRUMENT", 
            "SRL_PREDICATE", "SRL_GET_ROLE"
        ]

    @classmethod
    def from_passages(cls, passages: List[str], max_vocab: int = 200, s_dim: int = 20) -> "TextDomainConfig":
        doc_freq: Counter = Counter()
        for p in passages:
            doc_freq.update(set(tokenise(p)))
        vocab = [w for w, _ in doc_freq.most_common(max_vocab)]
        if not vocab: vocab = ["empty"] # fallback for empty corpus
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
        if norm > 0: concept_vec /= norm
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

    def expand_vocab(self, new_passages: List[str], max_new: int = 50) -> int:
        existing = set(self.concepts)
        doc_freq: Counter = Counter()
        for p in new_passages:
            doc_freq.update(set(tokenise(p)))
        candidates = [w for w, _ in doc_freq.most_common(max_new * 2) if w not in existing]
        added = candidates[:max_new]
        if not added: return 0
        
        n_docs = max(len(self._passages) + len(new_passages), 1)
        for w in added:
            self.concepts.append(w)
            self.idf[w] = math.log((n_docs + 1) / (doc_freq[w] + 1)) + 1.0
        self.DIM = len(self.concepts)
        self.m_dim = self.S_DIM + self.DIM + self.S_DIM
        return len(added)
