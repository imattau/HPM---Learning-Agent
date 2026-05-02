"""Differencing preprocessor for trend and acceleration extraction."""

from __future__ import annotations

from collections import deque
from numbers import Real
from typing import Any

import numpy as np

from ..adapter import AdapterPacket
from ..core import State
from .base import PreprocessedInput

_SENTINEL = float("nan")


class DifferencingPreprocessor:
    """Compute first and second differences of a numeric stream.

    Emits (value, d1, d2) where d1 = current - previous and d2 = d1 - previous_d1.
    Uses NaN for steps where there is not enough history, so consumers can detect
    warm-up observations without special-casing indices.

    Provides structural separation of level, trend, and acceleration — the three
    quantities that a hierarchical pattern learner needs to distinguish stationary
    signals from drifting or accelerating ones.
    """

    name: str = "differencing"
    requires: list[str] = []
    provides: list[str] = ["state"]

    def __init__(self, order: int = 2) -> None:
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        self.order = order
        self._history: deque[float] = deque(maxlen=order + 1)

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw = packet.raw
        if not isinstance(raw, Real):
            raise TypeError(f"DifferencingPreprocessor expects a real number, got {type(raw).__name__}")

        value = float(raw)
        self._history.append(value)

        arr = np.array(self._history, dtype=float)
        diffs = np.diff(arr, n=1)

        d1 = float(diffs[0]) if len(diffs) >= 1 else _SENTINEL
        d2 = _SENTINEL
        if self.order == 2:
            d2 = float(diffs[1] - diffs[0]) if len(diffs) >= 2 else _SENTINEL

        diff_value: tuple[float, ...]
        if self.order == 1:
            diff_value = (value, d1)
        else:
            diff_value = (value, d1, d2)

        context = dict(packet.context or {})
        context.update(
            {
                "domain": "differencing",
                "value_kind": "diff_tuple",
                "diff_order": self.order,
                "d1": d1,
                "d2": d2 if self.order == 2 else None,
                "warm": len(self._history) > self.order,
            }
        )
        packet.context = context
        packet.states.append(State(value=diff_value, context=context))
        packet.log(self.name, {"value": value, "d1": d1, "d2": d2}, role="adapter")
        return packet

    def preprocess(self, raw: Any, *, context: dict[str, Any] | None = None) -> PreprocessedInput:
        packet = AdapterPacket(raw=raw, context=dict(context or {}))
        packet = self.run(packet)
        state = packet.states[-1]
        return PreprocessedInput(state=state, context=dict(state.context), raw=raw, packet=packet)
