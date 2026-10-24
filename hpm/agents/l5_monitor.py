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

    Before calibration, strategy confidence uses the documented fixed
    thresholds. After fit_calibration(), the thresholds come from cumulative
    prequential residual quantiles, bounded by the defaults, and confidence
    interpolates from 0.9 (low surprise) to 0.3 (high surprise).
    """

    def __init__(
        self,
        theta_low: float = 0.2,
        theta_high: float = 0.5,
        direction_weight: float = 0.7,
        lower_quantile: float = 0.25,
        upper_quantile: float = 0.75,
        surprise_ewma_alpha: float = 0.25,
    ) -> None:
        if (not np.isfinite(theta_low) or not np.isfinite(theta_high)
                or theta_low < 0 or theta_high <= theta_low):
            raise ValueError("surprise thresholds must satisfy 0 <= theta_low < theta_high")
        if not np.isfinite(direction_weight) or not 0.0 <= direction_weight <= 1.0:
            raise ValueError("direction_weight must be between 0 and 1")
        if not np.isfinite(lower_quantile) or not np.isfinite(upper_quantile):
            raise ValueError("calibration quantiles must be finite")
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("calibration quantiles must satisfy 0 <= lower < upper <= 1")
        if not np.isfinite(surprise_ewma_alpha) or not 0.0 < surprise_ewma_alpha <= 1.0:
            raise ValueError("surprise_ewma_alpha must be in (0, 1]")
        self.theta_low = theta_low
        self.theta_high = theta_high
        self._base_theta_low = theta_low
        self._base_theta_high = theta_high
        self.direction_weight = direction_weight
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.surprise_ewma_alpha = float(surprise_ewma_alpha)
        self.calibrated = False
        self._calibration_surprises: list[float] = []
        self._surprises: list[float] = []
        self._surprise_ewma: float | None = None

    def _surprise(self, l4_pred: np.ndarray, actual_l3: np.ndarray) -> float:
        """Compute the normalized direction and magnitude residual."""
        if l4_pred.shape != actual_l3.shape:
            raise ValueError("L4 prediction and L3 target must have matching shapes")
        if not np.all(np.isfinite(l4_pred)) or not np.all(np.isfinite(actual_l3)):
            raise ValueError("L4 prediction and L3 target must be finite")
        cos_dist = _cosine_distance(l4_pred, actual_l3)
        norm_actual = float(np.linalg.norm(actual_l3))
        norm_pred = float(np.linalg.norm(l4_pred))
        mag_delta = abs(norm_pred - norm_actual) / max(norm_actual, 1e-8)
        return self.direction_weight * cos_dist + (1.0 - self.direction_weight) * mag_delta

    def fit_calibration(
        self, l4_predictions: list[np.ndarray], actual_l3_vectors: list[np.ndarray],
    ) -> dict:
        """Learn surprise thresholds from training-phase L4 prediction errors.

        Predictions must be generated before their corresponding L3 outcomes
        are observed (for example, sequential/rolling predictions). New
        residuals accumulate into persistent calibration history across rounds.
        Thresholds retain the default low/high anchors as conservative bounds,
        so a uniformly bad calibration set cannot label itself "high trust."
        """
        if len(l4_predictions) != len(actual_l3_vectors):
            raise ValueError("predictions and actual vectors must have equal length")
        if not l4_predictions:
            raise ValueError("at least one L4 prediction is required for calibration")
        new_surprises = [self._surprise(np.asarray(pred), np.asarray(actual))
                         for pred, actual in zip(l4_predictions, actual_l3_vectors)]
        if not np.all(np.isfinite(new_surprises)):
            raise ValueError("calibration residuals must be finite")
        self._calibration_surprises.extend(new_surprises)
        surprises = self._calibration_surprises
        low = min(self._base_theta_low, float(np.quantile(surprises, self.lower_quantile)))
        high = max(self._base_theta_high, float(np.quantile(surprises, self.upper_quantile)))
        # Keep a usable interpolation interval when calibration residuals tie.
        if high <= low:
            high = low + 1e-8
        self.theta_low, self.theta_high = low, high
        self.calibrated = True
        # Calibration inputs set thresholds but are not injected into task
        # history. Preserve any task events already observed prequentially;
        # reset() remains the task-boundary operation that clears those.
        return {"observations": len(new_surprises), "cumulative_observations": len(surprises),
                "lower_quantile": self.lower_quantile,
                "upper_quantile": self.upper_quantile, "theta_low": self.theta_low,
                "theta_high": self.theta_high,
                "mean_surprise": float(np.mean(new_surprises)),
                "cumulative_mean_surprise": float(np.mean(surprises))}

    def update(self, l4_pred: np.ndarray | None, actual_l3: np.ndarray) -> None:
        """Compute surprise and accumulate. No-op if l4_pred is None."""
        if l4_pred is None:
            return
        surprise = self._surprise(np.asarray(l4_pred), np.asarray(actual_l3))
        if not np.isfinite(surprise):
            raise ValueError("L4 prediction residual must be finite")
        self._surprises.append(surprise)
        if self._surprise_ewma is None:
            self._surprise_ewma = surprise
        else:
            alpha = self.surprise_ewma_alpha
            self._surprise_ewma = alpha * surprise + (1.0 - alpha) * self._surprise_ewma

    def strategic_confidence(self) -> float:
        """Return gamma from running mean surprise. Returns 1.0 if no data."""
        if not self._surprises:
            return 1.0
        s_bar = float(self._surprise_ewma if self._surprise_ewma is not None
                      else np.mean(self._surprises))
        if self.calibrated:
            progress = (s_bar - self.theta_low) / (self.theta_high - self.theta_low)
            return float(np.clip(0.9 - 0.6 * progress, 0.3, 0.9))
        if s_bar < self.theta_low:
            return 0.9
        elif s_bar > self.theta_high:
            return 0.3
        else:
            return 1.0 - s_bar

    def reset(self) -> None:
        """Clear task-local errors while retaining learned L5 calibration."""
        self._surprises = []
        self._surprise_ewma = None

    def state_dict(self) -> dict:
        """Return calibration and task-local state for checkpointing."""
        return {"theta_low": self.theta_low, "theta_high": self.theta_high,
                "base_theta_low": self._base_theta_low,
                "base_theta_high": self._base_theta_high,
                "direction_weight": self.direction_weight,
                "lower_quantile": self.lower_quantile,
                "upper_quantile": self.upper_quantile,
                "surprise_ewma_alpha": self.surprise_ewma_alpha,
                "calibrated": self.calibrated,
                "calibration_surprises": list(self._calibration_surprises),
                "surprises": list(self._surprises),
                "surprise_ewma": self._surprise_ewma}

    def load_state_dict(self, state: dict) -> None:
        """Restore state produced by state_dict()."""
        theta_low = float(state["theta_low"])
        theta_high = float(state["theta_high"])
        base_theta_low = float(state.get("base_theta_low", theta_low))
        base_theta_high = float(state.get("base_theta_high", theta_high))
        direction_weight = float(state["direction_weight"])
        lower_quantile = float(state.get("lower_quantile", 0.25))
        upper_quantile = float(state.get("upper_quantile", 0.75))
        surprise_ewma_alpha = float(state.get("surprise_ewma_alpha", 0.25))
        if (not np.isfinite(theta_low) or not np.isfinite(theta_high)
                or theta_low < 0 or theta_high <= theta_low):
            raise ValueError("surprise thresholds must satisfy 0 <= theta_low < theta_high")
        if (not np.isfinite(base_theta_low) or not np.isfinite(base_theta_high)
                or base_theta_low < 0 or base_theta_high <= base_theta_low):
            raise ValueError("base surprise thresholds must satisfy 0 <= low < high")
        if not np.isfinite(direction_weight) or not 0.0 <= direction_weight <= 1.0:
            raise ValueError("direction_weight must be between 0 and 1")
        if not np.isfinite(lower_quantile) or not np.isfinite(upper_quantile):
            raise ValueError("calibration quantiles must be finite")
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("calibration quantiles must satisfy 0 <= lower < upper <= 1")
        if not np.isfinite(surprise_ewma_alpha) or not 0.0 < surprise_ewma_alpha <= 1.0:
            raise ValueError("surprise_ewma_alpha must be in (0, 1]")
        self.theta_low, self.theta_high = theta_low, theta_high
        self._base_theta_low, self._base_theta_high = base_theta_low, base_theta_high
        self.direction_weight = direction_weight
        self.lower_quantile, self.upper_quantile = lower_quantile, upper_quantile
        self.surprise_ewma_alpha = surprise_ewma_alpha
        self.calibrated = bool(state.get("calibrated", False))
        self._calibration_surprises = [float(value) for value in state.get("calibration_surprises", [])]
        self._surprises = [float(value) for value in state.get("surprises", [])]
        ewma = state.get("surprise_ewma")
        self._surprise_ewma = None if ewma is None else float(ewma)
        if (any(not np.isfinite(value) or value < 0
                for value in (*self._calibration_surprises, *self._surprises))):
            raise ValueError("stored surprise values must be finite and nonnegative")
        if self._surprise_ewma is not None and (
                not np.isfinite(self._surprise_ewma) or self._surprise_ewma < 0):
            raise ValueError("stored surprise EWMA must be finite and nonnegative")
        if self.calibrated and not self._calibration_surprises:
            raise ValueError("calibrated state must include calibration residuals")
