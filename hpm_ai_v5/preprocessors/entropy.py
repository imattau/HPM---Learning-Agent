"""Entropy and surprise preprocessor for novelty detection."""

from __future__ import annotations

from collections import deque
from numbers import Real
from typing import Any

import numpy as np

from ..adapter import AdapterPacket
from ..core import State
from .base import PreprocessedInput


class EntropyPreprocessor:
    """Compute Shannon entropy and point-level surprise over a rolling window.

    Entropy captures population-level uncertainty: near-zero on monotonic or
    constant streams, higher on periodic streams (mass spread over k symbols),
    highest on structurally novel or random streams.

    Surprise captures how unexpected the current observation is relative to
    the window's recent distribution (z-score magnitude, clipped to [0, 1]).

    Together these implement HPM's pattern gatekeeper signal: high entropy +
    high surprise means the pattern field is unstable and worth attending to.
    """

    name: str = "entropy"
    requires: list[str] = []
    provides: list[str] = ["state"]

    def __init__(
        self,
        window_size: int = 12,
        n_bins: int = 8,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        self.window_size = window_size
        self.n_bins = n_bins
        self.window: deque[float] = deque(maxlen=window_size)

    def _shannon_entropy(self, arr: np.ndarray) -> float:
        if len(arr) < 2:
            return 0.0
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-10:
            return 0.0
        counts, _ = np.histogram(arr, bins=self.n_bins, range=(lo, hi))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        raw = float(-np.sum(probs * np.log2(probs)))
        # normalise to [0, 1] by max possible entropy (log2 of n_bins)
        return raw / np.log2(self.n_bins)

    def _surprise(self, arr: np.ndarray, current: float) -> float:
        if len(arr) < 2:
            return 0.0
        std = float(arr.std())
        if std < 1e-10:
            return 0.0
        z = abs(current - float(arr.mean())) / std
        # sigmoid-style clip: tanh maps z-score to (0, 1)
        return float(np.tanh(z / 2.0))

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw = packet.raw
        if not isinstance(raw, Real):
            raise TypeError(f"EntropyPreprocessor expects a real number, got {type(raw).__name__}")

        current = float(raw)
        self.window.append(current)
        arr = np.array(self.window, dtype=float)

        entropy = self._shannon_entropy(arr)
        surprise = self._surprise(arr[:-1] if len(arr) > 1 else arr, current)
        state_value = (entropy, surprise)

        context = dict(packet.context or {})
        context.update(
            {
                "domain": "entropy",
                "value_kind": "entropy_tuple",
                "window_size": self.window_size,
                "n_bins": self.n_bins,
                "current_window_len": len(self.window),
                "entropy": entropy,
                "surprise": surprise,
                "warm": len(self.window) >= self.window_size,
            }
        )
        packet.context = context
        packet.states.append(State(value=state_value, context=context))
        packet.log(self.name, {"entropy": entropy, "surprise": surprise}, role="adapter")
        return packet

    def preprocess(self, raw: Any, *, context: dict[str, Any] | None = None) -> PreprocessedInput:
        packet = AdapterPacket(raw=raw, context=dict(context or {}))
        packet = self.run(packet)
        state = packet.states[-1]
        return PreprocessedInput(state=state, context=dict(state.context), raw=raw, packet=packet)
