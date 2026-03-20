from typing import Protocol, Iterator, runtime_checkable
import hashlib
import numpy as np


def hash_vectorise(text: str, dim: int = 32) -> np.ndarray:
    """
    Hash-trick text vectoriser. Maps words to a fixed-dim float array.
    Uses hashlib.md5 for cross-process determinism — Python's hash() is
    randomised per process (PYTHONHASHSEED), md5 is stable across runs.
    Returns a normalised word-frequency vector.
    """
    vec = np.zeros(dim)
    words = text.lower().split()
    for word in words:
        digest = hashlib.md5(word.encode()).digest()
        idx = int.from_bytes(digest[:4], 'little') % dim
        vec[idx] += 1.0
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


@runtime_checkable
class ExternalSubstrate(Protocol):
    def fetch(self, query: str) -> list[np.ndarray]: ...
    def field_frequency(self, pattern) -> float: ...
    def stream(self) -> Iterator[np.ndarray]: ...
