"""Rolling statistics preprocessing over a sliding window."""

from __future__ import annotations

from collections import deque
from numbers import Real
from typing import Any

import numpy as np

from ..adapter import AdapterPacket
from ..core import State
from .base import PreprocessedInput


class RollingStatsPreprocessor:
    """Compute mean, std, min, max over a sliding window of recent observations.

    Emits a fused tuple (mean, std, min, max, range) as the state value,
    giving downstream pattern learners a statistical fingerprint of recent
    history rather than just the raw value.
    """

    name: str = "rolling_stats"
    requires: list[str] = []
    provides: list[str] = ["state"]

    def __init__(self, window_size: int = 4) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size = window_size
        self.window: deque[float] = deque(maxlen=window_size)

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw = packet.raw
        if not isinstance(raw, Real):
            raise TypeError(f"RollingStatsPreprocessor expects a real number, got {type(raw).__name__}")

        self.window.append(float(raw))
        arr = np.array(self.window, dtype=float)

        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        minimum = float(np.min(arr))
        maximum = float(np.max(arr))
        value_range = maximum - minimum
        stats_value = (mean, std, minimum, maximum, value_range)

        context = dict(packet.context or {})
        context.update(
            {
                "domain": "rolling_stats",
                "value_kind": "stats_tuple",
                "window_size": self.window_size,
                "current_window_len": len(self.window),
                "stats_mean": mean,
                "stats_std": std,
                "stats_min": minimum,
                "stats_max": maximum,
                "stats_range": value_range,
            }
        )
        packet.context = context
        packet.states.append(State(value=stats_value, context=context))
        packet.log(
            self.name,
            {"stats": {"mean": mean, "std": std, "min": minimum, "max": maximum, "range": value_range}},
            role="adapter",
        )
        return packet

    def preprocess(self, raw: Any, *, context: dict[str, Any] | None = None) -> PreprocessedInput:
        packet = AdapterPacket(raw=raw, context=dict(context or {}))
        packet = self.run(packet)
        state = packet.states[-1]
        return PreprocessedInput(state=state, context=dict(state.context), raw=raw, packet=packet)
