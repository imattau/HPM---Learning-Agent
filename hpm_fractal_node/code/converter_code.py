"""
ConverterCode — encodes raw query strings from QueryStdlib into D-dimensional vectors.

Two encoding paths:
  - "sig: ..." strings: bag-of-tokens encoding = (1/N) * sum(one_hot(t) for t in tokens)
  - Context window strings: split into 4 tokens, call compose_context_node(l2, l1, r1, r2)
"""
from __future__ import annotations

import numpy as np

from hfn.converter import Converter
from hpm_fractal_node.code.code_loader import (
    VOCAB_INDEX, VOCAB_SIZE, compose_context_node,
)


class ConverterCode(Converter):
    """
    Encode raw query strings (from QueryStdlib) into D-dimensional vectors.

    For "sig: ..." strings:
        bag-of-tokens = mean of one-hot vectors for each known token in the signature.
        Unknown tokens ("<unk>" or not in VOCAB) are skipped.

    For context window strings ("l2 l1 r1 r2"):
        calls compose_context_node(l2, l1, r1, r2).
        "<unk>" tokens contribute zero weight (handled by compose_context_node).

    Parameters
    ----------
    raw : str
        A raw string from QueryStdlib.fetch().
    D : int
        Target dimensionality — must equal VOCAB_SIZE (70).

    Returns
    -------
    list[np.ndarray]
        One D-dimensional array per successfully encoded string.
        Returns empty list if the string cannot be encoded.
    """

    def encode(self, raw: str, D: int) -> list[np.ndarray]:
        """Encode a raw string into a list of D-dimensional arrays."""
        if raw.startswith("sig: "):
            return self._encode_sig(raw[5:], D)  # strip "sig: " prefix
        else:
            return self._encode_context(raw, D)

    def _encode_sig(self, sig_body: str, D: int) -> list[np.ndarray]:
        """Bag-of-tokens encoding for signature strings."""
        tokens = sig_body.split()
        vecs = []
        for token in tokens:
            if token == "<unk>" or token not in VOCAB_INDEX:
                continue
            idx = VOCAB_INDEX[token]
            vec = np.zeros(D, dtype=np.float64)
            vec[idx] = 1.0
            vecs.append(vec)
        if not vecs:
            return []
        result = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float64)
        return [result]

    def _encode_context(self, raw: str, D: int) -> list[np.ndarray]:
        """Context window encoding for 4-token strings."""
        parts = raw.split()
        if len(parts) != 4:
            return []
        l2, l1, r1, r2 = parts
        vec = compose_context_node(l2, l1, r1, r2)
        if vec.shape[0] != D:
            return []
        return [vec]
