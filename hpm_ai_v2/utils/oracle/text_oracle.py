"""Oracle that evaluates text passage retrieval quality via cosine similarity."""
from __future__ import annotations
from typing import Any, List, Optional
import numpy as np
from hpm_ai_v2.utils.oracle.base import BaseOracle
from hpm_ai_v2.domains.text_domain import TextDomainConfig


class TextOracle(BaseOracle):
    def __init__(self, config: TextDomainConfig) -> None:
        self.config = config

    def compute_state(
        self,
        outputs: List[Any],
        errors: List[Optional[str]],
        code: str = "",
        inputs: Optional[List[Any]] = None,
    ) -> np.ndarray:
        state = np.zeros(self.config.S_DIM)
        if not outputs or not isinstance(outputs[0], str):
            return state
        query_mu = self.config.encode_passage(outputs[0])
        query_vec = query_mu[self.config.S_DIM: self.config.S_DIM + self.config.DIM]
        if not self.config._passage_vecs:
            return state
        sims = []
        for pvec in self.config._passage_vecs:
            pvec_concept = pvec[self.config.S_DIM: self.config.S_DIM + self.config.DIM]
            denom = (np.linalg.norm(query_vec) * np.linalg.norm(pvec_concept)) + 1e-9
            sims.append(float(np.dot(query_vec, pvec_concept) / denom))
        state[0] = max(sims) if sims else 0.0
        state[1] = float(np.mean(sims)) if sims else 0.0
        return state
