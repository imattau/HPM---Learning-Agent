"""
Converter — abstract base class for encoding raw query results into HFN observations.

A Converter translates raw observation strings (returned by a Query) into
D-dimensional numpy arrays that the Observer can use to create new HFN nodes.
"""

from __future__ import annotations

import numpy as np


class Converter:
    """
    Abstract base for raw-string-to-vector encoders.

    Subclasses override encode() to implement domain-specific encoding.
    Attempting to use the base class encode() directly raises NotImplementedError.

    Parameters
    ----------
    raw : str
        A raw observation string from a Query.fetch() call.
    D : int
        Target dimensionality for the output vectors.

    Returns
    -------
    list[np.ndarray]
        One D-dimensional observation array per successfully encoded string.
        May return fewer items than len(raw_list) if some strings cannot be encoded.
    """

    def encode(self, raw: str, D: int) -> list[np.ndarray]:
        """Encode a raw string into a list of D-dimensional arrays.

        Must be overridden by subclasses.
        """
        raise NotImplementedError("Converter.encode() must be implemented by subclasses")
