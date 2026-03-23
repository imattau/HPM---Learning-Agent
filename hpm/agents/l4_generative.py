"""L4GenerativeHead: Online ridge regressor mapping L2 vectors to predicted L3 vectors."""
from __future__ import annotations

import numpy as np


class L4GenerativeHead:
    """L4 — Online ridge regressor mapping L2 vectors to predicted L3 vectors.

    Accumulates (L2, L3) training pairs during the training phase of each task,
    then solves W = (X^T X + alpha*I)^{-1} X^T Y at test time via np.linalg.solve.
    Stateless between tasks: call reset() at each task boundary.
    """

    def __init__(
        self,
        feature_dim_in: int,
        feature_dim_out: int,
        alpha: float = 0.01,
    ) -> None:
        self.feature_dim_in = feature_dim_in
        self.feature_dim_out = feature_dim_out
        self.alpha = alpha
        self._X: list[np.ndarray] = []
        self._Y: list[np.ndarray] = []
        self._W: np.ndarray | None = None

    def accumulate(self, l2_vec: np.ndarray, actual_l3_vec: np.ndarray) -> None:
        """Store a (L2, L3) training pair."""
        self._X.append(l2_vec.copy())
        self._Y.append(actual_l3_vec.copy())

    def fit(self) -> None:
        """Solve ridge regression. No-op if fewer than 2 pairs accumulated."""
        if len(self._X) < 2:
            return
        X = np.stack(self._X)          # (N, d_in)
        Y = np.stack(self._Y)          # (N, d_out)
        d = X.shape[1]
        A = X.T @ X + self.alpha * np.eye(d)
        B = X.T @ Y
        self._W = np.linalg.solve(A, B)  # (d_in, d_out)

    def predict(self, l2_vec: np.ndarray) -> np.ndarray | None:
        """Return l2_vec @ W, or None if not yet fitted."""
        if self._W is None:
            return None
        return l2_vec @ self._W

    def reset(self) -> None:
        """Clear accumulated data and fitted weights (call at task boundary)."""
        self._X = []
        self._Y = []
        self._W = None
