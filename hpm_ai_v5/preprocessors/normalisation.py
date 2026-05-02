"""Normalisation preprocessing — makes numeric streams scale-invariant."""

from __future__ import annotations

from collections import deque
from numbers import Real
from typing import Any

import numpy as np

from ..adapter import AdapterPacket
from ..core import State
from .base import PreprocessedInput


class NormalisationPreprocessor:
    """Normalise a numeric stream so structurally identical streams at different magnitudes
    produce the same downstream state representation.

    Two modes:
    - ``"zscore"`` (default): ``(value - mean) / std`` over a rolling window.
    - ``"minmax"``: ``(value - min) / (max - min)`` over a rolling window, clamped to [0, 1].

    During warm-up (window not yet full) normalisation still uses available data.
    """

    name: str = "normalisation"
    requires: list[str] = []
    provides: list[str] = ["state"]

    def __init__(self, mode: str = "zscore", window_size: int = 12) -> None:
        if mode not in ("zscore", "minmax"):
            raise ValueError(f"mode must be 'zscore' or 'minmax', got {mode!r}")
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.mode = mode
        self.window_size = window_size
        self.window: deque[float] = deque(maxlen=window_size)

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw = packet.raw
        if not isinstance(raw, Real):
            raise TypeError(
                f"NormalisationPreprocessor expects a real number, got {type(raw).__name__}"
            )

        self.window.append(float(raw))
        arr = np.array(self.window, dtype=float)
        warm = len(self.window) == self.window_size

        if self.mode == "zscore":
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=0))
            normalised = 0.0 if std < 1e-10 else (float(raw) - mean) / std
        else:  # minmax
            minimum = float(np.min(arr))
            maximum = float(np.max(arr))
            value_range = maximum - minimum
            if value_range < 1e-10:
                normalised = 0.0
            else:
                normalised = (float(raw) - minimum) / value_range
                normalised = max(0.0, min(1.0, normalised))

        context = dict(packet.context or {})
        context.update(
            {
                "domain": "normalisation",
                "value_kind": "normalised_scalar",
                "mode": self.mode,
                "window_size": self.window_size,
                "current_window_len": len(self.window),
                "normalised_value": normalised,
                "raw_value": float(raw),
                "warm": warm,
            }
        )
        packet.context = context
        packet.states.append(State(value=normalised, context=context))
        packet.log(
            self.name,
            {"normalised_value": normalised, "raw_value": float(raw), "mode": self.mode},
            role="adapter",
        )
        return packet

    def preprocess(self, raw: Any, *, context: dict[str, Any] | None = None) -> PreprocessedInput:
        packet = AdapterPacket(raw=raw, context=dict(context or {}))
        packet = self.run(packet)
        state = packet.states[-1]
        return PreprocessedInput(state=state, context=dict(state.context), raw=raw, packet=packet)
