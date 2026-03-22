"""ArchiveForecaster: ranks CandidateArchive entries by Pattern Fingerprint NLL.

Extracted from ContextualPatternStore as part of Phase 3 refactor.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hpm.store.archive_librarian import CandidateArchive


@dataclass
class RankedCandidate:
    candidate: CandidateArchive
    mean_nll: float


class ArchiveForecaster:
    """Ranks archive candidates by mean NLL over a set of observations."""

    def rank(self, candidates: list[CandidateArchive], obs: list[np.ndarray],
             nll_threshold: float = 50.0) -> list[RankedCandidate]:
        """Load patterns from each candidate, compute mean NLL over obs, filter by threshold, sort ascending.

        When obs is empty, return all candidates with mean_nll=0.0 (no NLL scoring possible).
        """
        if not candidates:
            return []

        # No observations to filter on: return all candidates unranked
        if not obs:
            return [RankedCandidate(candidate=c, mean_nll=0.0) for c in candidates]

        results = []
        for candidate in candidates:
            archive_path = Path(candidate.archive_path)
            if not archive_path.exists():
                continue
            try:
                with open(archive_path, "rb") as f:
                    records = pickle.load(f)
            except Exception:
                continue
            patterns = [p for p, _w, _aid in records]
            if not patterns:
                continue
            mean_nll = self._mean_nll(patterns, obs)
            if mean_nll < nll_threshold:
                results.append(RankedCandidate(candidate=candidate, mean_nll=mean_nll))

        results.sort(key=lambda r: r.mean_nll)
        return results

    def _mean_nll(self, patterns, obs_list: list) -> float:
        nlls = []
        for obs in obs_list:
            for p in patterns:
                nlls.append(float(p.log_prob(obs)))
        return float(np.mean(nlls)) if nlls else float("inf")
