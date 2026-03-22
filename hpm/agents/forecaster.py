"""Forecaster: Pattern Fingerprint NLL selection logic.

Extracted from ContextualPatternStore (Phase 3 refactor).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

from hpm.agents.librarian import CandidateArchive


@dataclass
class RankedCandidate:
    candidate: CandidateArchive
    mean_nll: float


class Forecaster:
    """Stateless helper: rank archive candidates by mean NLL over observations."""

    def rank(
        self,
        candidates: list[CandidateArchive],
        obs: list[np.ndarray],
        nll_threshold: float = 50.0,
    ) -> list[RankedCandidate]:
        """Load each candidate's patterns, compute mean NLL over obs, filter by threshold, sort ascending.

        When obs is empty, all candidates pass through with mean_nll=0.0 in
        original order (no NLL scoring is possible without observations).

        Returns a list of RankedCandidate sorted by mean_nll ascending (best first),
        containing only those below nll_threshold.
        """
        if not candidates:
            return []

        # No observations: skip NLL scoring, return all candidates with nll=0
        if not obs:
            return [
                RankedCandidate(candidate=c, mean_nll=0.0) for c in candidates
            ]

        ranked: list[RankedCandidate] = []
        for candidate in candidates:
            archive_path = candidate.archive_path
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
                ranked.append(RankedCandidate(candidate=candidate, mean_nll=mean_nll))

        ranked.sort(key=lambda rc: rc.mean_nll)
        return ranked

    def _mean_nll(self, patterns, obs_list: list) -> float:
        nlls = []
        for o in obs_list:
            for p in patterns:
                nlls.append(float(p.log_prob(o)))
        return float(np.mean(nlls)) if nlls else float("inf")
