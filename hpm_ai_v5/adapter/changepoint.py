"""Changepoint / regime-shift detector adapter."""

from __future__ import annotations

import math
from collections import deque
from numbers import Real

import numpy as np

from ..core.state import State
from .packet import AdapterPacket


class ChangepointAdapter:
    """Detect structural regime shifts by comparing two halves of a rolling window.

    When a shift is detected ``regime_changed=True`` is emitted into
    ``packet.context`` so the caller can trigger episode boundary handling.
    """

    name = "changepoint"
    requires: list[str] = []
    provides: list[str] = ["state"]

    def __init__(
        self,
        window_size: int = 16,
        threshold: float = 1.5,
        cooldown: int = 4,
    ) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.cooldown = cooldown

        self.window: deque[float] = deque(maxlen=window_size)
        self._cooldown_remaining: int = 0
        self._regime_count: int = 0

    def _shift_score(self) -> float:
        half = self.window_size // 2
        old = np.array(list(self.window)[:half], dtype=float)
        new = np.array(list(self.window)[half:], dtype=float)

        mean_old, mean_new = old.mean(), new.mean()
        var_old, var_new = old.var(), new.var()

        std_pooled = math.sqrt((var_old + var_new) / 2.0)
        mean_shift_norm = abs(mean_new - mean_old) / (std_pooled + 1e-10)

        var_ratio = max(var_new, var_old) / (min(var_new, var_old) + 1e-10)
        score = 0.6 * mean_shift_norm + 0.4 * math.log(var_ratio + 1)
        return score

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw = packet.raw
        if not isinstance(raw, Real):
            raise TypeError(f"ChangepointAdapter requires a real numeric raw value, got {type(raw)}")

        self.window.append(float(raw))

        warm = len(self.window) >= self.window_size

        if not warm:
            shift_score = 0.0
            regime_changed = False
        else:
            shift_score = self._shift_score()

            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                regime_changed = False
            elif shift_score >= self.threshold:
                regime_changed = True
                self._regime_count += 1
                self._cooldown_remaining = self.cooldown
            else:
                regime_changed = False

        packet.context["regime_changed"] = regime_changed
        packet.context["shift_score"] = round(shift_score, 4)
        packet.context["regime_count"] = self._regime_count
        packet.context["warm"] = warm
        packet.context["changepoint_threshold"] = self.threshold

        packet.states.append(State(value=(float(regime_changed), shift_score)))
        packet.log(self.name, {"regime_changed": regime_changed, "shift_score": round(shift_score, 4)})

        return packet

    def reset(self) -> None:
        """Clear window and counters — call after manual episode boundary."""
        self.window.clear()
        self._cooldown_remaining = 0
