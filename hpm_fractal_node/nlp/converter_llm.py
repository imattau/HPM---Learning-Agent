"""
ConverterLLM — encodes QueryLLM 3-group responses into HFN observation vectors.

Each input line has the form:
    similar: word1, word2, word3
    related: word1, word2, word3
    context: word1, word2, word3

Each group is encoded as a normalised bag-of-words vector over the NLP
vocabulary (D=107).  Words not present in the vocabulary are skipped.
Groups that produce a zero vector are omitted from the output.

Returns up to 3 vectors per call (one per valid group).
"""
from __future__ import annotations

import numpy as np

from hfn.converter import Converter
from hpm_fractal_node.nlp.nlp_loader import VOCAB_INDEX, VOCAB_SIZE


def _bag_of_words(words: list[str]) -> np.ndarray:
    """Sum of one-hot vectors for words present in vocabulary, L2-normalised."""
    vec = np.zeros(VOCAB_SIZE, dtype=np.float64)
    for w in words:
        idx = VOCAB_INDEX.get(w.lower().strip())
        if idx is not None:
            vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _parse_group(line: str) -> list[str]:
    """Parse 'label: word1, word2, word3' → ['word1', 'word2', 'word3']."""
    if ":" not in line:
        return []
    _, rest = line.split(":", 1)
    return [w.strip().lower() for w in rest.split(",") if w.strip()]


class ConverterLLM(Converter):
    """
    Converts a QueryLLM response line into bag-of-words HFN vectors.

    Parameters
    ----------
    raw : str
        A single response line from QueryLLM (e.g. "similar: cat, bird, dog").
    D : int
        Target dimensionality — must match VOCAB_SIZE (107).

    Returns
    -------
    list[np.ndarray]
        Zero or one D-dimensional vector.  Empty if no vocab words matched.
    """

    def encode(self, raw: str, D: int) -> list[np.ndarray]:
        words = _parse_group(raw)
        if not words:
            return []
        vec = _bag_of_words(words)
        if np.all(vec == 0):
            return []
        # Resize to D if needed (handles D != VOCAB_SIZE gracefully)
        if len(vec) != D:
            resized = np.zeros(D, dtype=np.float64)
            n = min(len(vec), D)
            resized[:n] = vec[:n]
            norm = np.linalg.norm(resized)
            if norm > 0:
                resized /= norm
            return [resized]
        return [vec]
