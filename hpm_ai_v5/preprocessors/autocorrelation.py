"""Autocorrelation preprocessor for periodicity detection."""

from __future__ import annotations

from collections import deque
from numbers import Real
from typing import Any

import numpy as np

from ..adapter import AdapterPacket
from ..core import State
from .base import PreprocessedInput


class AutocorrelationPreprocessor:
    """Compute lagged autocorrelation over a rolling window.

    Emits a tuple of ACF values at lags 1..max_lag, plus the dominant period
    (lag with highest ACF, or 0 if no significant periodicity is found).

    This gives downstream selectors a structural fingerprint of the stream —
    high ACF at lag k means the sequence repeats with period k — without
    needing to race pipelines to discover structure empirically.
    """

    name: str = "autocorrelation"
    requires: list[str] = []
    provides: list[str] = ["state"]

    def __init__(
        self,
        window_size: int = 12,
        max_lag: int = 4,
        significance_threshold: float = 0.4,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if max_lag < 1:
            raise ValueError("max_lag must be at least 1")
        if max_lag >= window_size:
            raise ValueError("max_lag must be less than window_size")
        self.window_size = window_size
        self.max_lag = max_lag
        self.significance_threshold = significance_threshold
        self.window: deque[float] = deque(maxlen=window_size)

    def _compute_acf(self, arr: np.ndarray) -> np.ndarray:
        n = len(arr)
        if n < 2:
            return np.zeros(self.max_lag)
        mean = arr.mean()
        demeaned = arr - mean
        variance = (demeaned ** 2).mean()
        if variance < 1e-10:
            return np.zeros(self.max_lag)
        acf = np.array([
            float(np.mean(demeaned[:n - lag] * demeaned[lag:])) / variance
            if n > lag else 0.0
            for lag in range(1, self.max_lag + 1)
        ])
        return np.clip(acf, -1.0, 1.0)

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw = packet.raw
        if not isinstance(raw, Real):
            raise TypeError(f"AutocorrelationPreprocessor expects a real number, got {type(raw).__name__}")

        self.window.append(float(raw))
        arr = np.array(self.window, dtype=float)
        acf = self._compute_acf(arr)

        # dominant period: lag with highest ACF above threshold
        significant = [(lag + 1, v) for lag, v in enumerate(acf) if v >= self.significance_threshold]
        dominant_period = max(significant, key=lambda x: x[1])[0] if significant else 0

        acf_value: tuple[float, ...] = (*tuple(float(v) for v in acf), float(dominant_period))

        context = dict(packet.context or {})
        context.update(
            {
                "domain": "autocorrelation",
                "value_kind": "acf_tuple",
                "window_size": self.window_size,
                "max_lag": self.max_lag,
                "current_window_len": len(self.window),
                "dominant_period": dominant_period,
                "acf_lags": {f"lag_{lag + 1}": float(v) for lag, v in enumerate(acf)},
                "warm": len(self.window) >= self.window_size,
            }
        )
        packet.context = context
        packet.states.append(State(value=acf_value, context=context))
        packet.log(
            self.name,
            {"acf": list(float(v) for v in acf), "dominant_period": dominant_period},
            role="adapter",
        )
        return packet

    def preprocess(self, raw: Any, *, context: dict[str, Any] | None = None) -> PreprocessedInput:
        packet = AdapterPacket(raw=raw, context=dict(context or {}))
        packet = self.run(packet)
        state = packet.states[-1]
        return PreprocessedInput(state=state, context=dict(state.context), raw=raw, packet=packet)
