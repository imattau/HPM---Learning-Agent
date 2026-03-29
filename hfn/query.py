"""
Query — abstract base class for gap-driven knowledge queries.

A Query translates a coverage-gap signal (gap_mu vector) into a list of
raw observation strings that the Observer can encode and inject into the
Forest as new HFN nodes.
"""

from __future__ import annotations

import numpy as np


class Query:
    """
    Abstract base for gap queries.

    Subclasses override fetch() to retrieve domain-specific knowledge for
    the pattern identified by gap_mu.

    Parameters
    ----------
    gap_mu : np.ndarray
        The mean vector of the coverage gap — used to identify what token /
        concept / pattern the gap corresponds to.
    context : object, optional
        Additional context (e.g. the current explanation tree).

    Returns
    -------
    list[str]
        Raw observation strings.  Strings prefixed with "sig: " are treated
        as signature/definition strings by the Observer.
    """

    def fetch(self, gap_mu: np.ndarray, context=None) -> list[str]:
        """Return raw observation strings for the given gap.  Default: empty list."""
        return []
