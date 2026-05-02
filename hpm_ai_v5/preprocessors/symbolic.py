"""Symbolic discretiser: maps continuous numeric values to discrete integer symbols."""

from __future__ import annotations

from collections import deque
from numbers import Real
from typing import Any

import numpy as np

from ..adapter import AdapterPacket
from ..core import State
from .base import PreprocessedInput

_MODES = {"uniform", "quantile", "adaptive"}


class SymbolicDiscretiser:
    """Map continuous numeric values to discrete integer symbols.

    Bridges numeric streams into symbolic pattern learning. Structurally
    identical streams at different scales produce the same symbol sequence
    when using adaptive modes.

    Modes
    -----
    ``"uniform"`` (default)
        Fixed equal-width bins over a fixed ``value_range``. Deterministic
        and scale-dependent. Good for known-range domains (e.g. ARC grid 0-9).
    ``"quantile"``
        Adaptive — bins based on empirical quantiles of a rolling window.
        Scale-invariant. Uses ``numpy.percentile``.
    ``"adaptive"``
        Adaptive — equal-width bins over rolling ``(min, max)``.
        Scale-invariant but sensitive to outliers.
    """

    name: str = "symbolic"
    requires: list[str] = []
    provides: list[str] = ["state"]

    def __init__(
        self,
        n_symbols: int = 8,
        mode: str = "uniform",
        value_range: tuple[float, float] = (0.0, 1.0),
        window_size: int = 12,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
        if n_symbols < 1:
            raise ValueError("n_symbols must be at least 1")
        if window_size < 1:
            raise ValueError("window_size must be at least 1")

        self.n_symbols = n_symbols
        self.mode = mode
        self.value_range = value_range
        self.window_size = window_size
        self.window: deque[float] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discretise(self, value: float) -> int:
        """Return symbol index in ``[0, n_symbols - 1]``."""
        if self.mode == "uniform":
            lo, hi = self.value_range
            if hi == lo:
                return 0
            normalised = (value - lo) / (hi - lo)
        elif self.mode == "quantile":
            arr = np.array(self.window, dtype=float)
            percentiles = np.linspace(0.0, 100.0, self.n_symbols + 1)
            edges = np.percentile(arr, percentiles)
            # searchsorted on interior edges
            idx = int(np.searchsorted(edges[1:-1], value, side="right"))
            return min(idx, self.n_symbols - 1)
        else:  # adaptive
            arr = np.array(self.window, dtype=float)
            lo, hi = float(arr.min()), float(arr.max())
            if hi == lo:
                return 0
            normalised = (value - lo) / (hi - lo)

        # uniform / adaptive share the clip-and-floor path
        symbol = int(normalised * self.n_symbols)
        return max(0, min(symbol, self.n_symbols - 1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw = packet.raw
        if not isinstance(raw, Real):
            raise TypeError(
                f"SymbolicDiscretiser expects a real number, got {type(raw).__name__}"
            )

        value = float(raw)
        self.window.append(value)

        symbol = self._discretise(value)
        warm = len(self.window) >= self.window_size

        context = dict(packet.context or {})
        context.update(
            {
                "domain": "symbolic",
                "value_kind": "symbol",
                "mode": self.mode,
                "n_symbols": self.n_symbols,
                "symbol": symbol,
                "raw_value": value,
                "warm": warm,
            }
        )
        packet.context = context
        packet.states.append(State(value=symbol, context=context))
        packet.log(
            self.name,
            {"symbol": symbol, "raw_value": value, "mode": self.mode, "warm": warm},
            role="adapter",
        )
        return packet

    def preprocess(self, raw: Any, *, context: dict[str, Any] | None = None) -> PreprocessedInput:
        packet = AdapterPacket(raw=raw, context=dict(context or {}))
        packet = self.run(packet)
        state = packet.states[-1]
        return PreprocessedInput(state=state, context=dict(state.context), raw=raw, packet=packet)
