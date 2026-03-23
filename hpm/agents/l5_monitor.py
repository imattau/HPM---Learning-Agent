"""L5MetaMonitor: Metacognitive monitor tracking L4 prediction surprise."""
from __future__ import annotations

import numpy as np


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 1]. Returns 1.0 if either vector is near-zero."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(1.0 - cos_sim, 0.0, 2.0))


class L5MetaMonitor:
    """L5 — Metacognitive monitor that tracks L4 prediction surprise.

    Computes a running mean of per-step surprise values and outputs a
    strategic_confidence scalar gamma that gates scoring interpolation in
    StructuredOrchestrator. Call reset() at each task boundary.

    Surprise metric:
        surprise_t = 0.7 * cos_dist(pred, actual) + 0.3 * |norm(pred)-norm(actual)| / max(norm(actual), 1e-8)

    Strategy selection from running mean S_bar:
        S_bar < 0.2     -> Exploit (trust L4)    -> gamma = 0.9
        0.2 <= S_bar <= 0.5 -> Neutral           -> gamma = 1 - S_bar
        S_bar > 0.5     -> Explore (override L4) -> gamma = 0.3
    """

    def __init__(
        self,
        theta_low: float = 0.2,
        theta_high: float = 0.5,
        direction_weight: float = 0.7,
    ) -> None:
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.direction_weight = direction_weight
        self._surprises: list[float] = []

    def update(self, l4_pred: np.ndarray | None, actual_l3: np.ndarray) -> None:
        """Compute surprise and accumulate. No-op if l4_pred is None."""
        if l4_pred is None:
            return
        cos_dist = _cosine_distance(l4_pred, actual_l3)
        norm_actual = float(np.linalg.norm(actual_l3))
        norm_pred = float(np.linalg.norm(l4_pred))
        mag_delta = abs(norm_pred - norm_actual) / max(norm_actual, 1e-8)
        surprise = self.direction_weight * cos_dist + (1.0 - self.direction_weight) * mag_delta
        self._surprises.append(surprise)

    def strategic_confidence(self) -> float:
        """Return gamma from running mean surprise. Returns 1.0 if no data."""
        if not self._surprises:
            return 1.0
        s_bar = float(np.mean(self._surprises))
        if s_bar < self.theta_low:
            return 0.9
        elif s_bar > self.theta_high:
            return 0.3
        else:
            return 1.0 - s_bar

    def reset(self) -> None:
        """Clear accumulated surprises (call at task boundary)."""
        self._surprises = []
