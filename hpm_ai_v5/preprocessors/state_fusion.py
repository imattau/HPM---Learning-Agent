"""Fusion preprocessing for automatic adapter composition benchmarks."""

from __future__ import annotations

from collections import Counter, defaultdict
from numbers import Real
from typing import Any

from ..adapter import AdapterPacket
from ..core import State
from .base import PreprocessedInput
from .prefix_buffer import _to_float_tuple


class StateFusionPreprocessor:
    """Fuse scalar value, local delta, and prefix history into one structured state."""

    name: str = "state_fusion"
    requires: list[str] = ["numeric", "prefix_buffer"]
    provides: list[str] = ["state"]

    def __init__(self) -> None:
        self.transition_memory: defaultdict[tuple[float, ...], Counter] = defaultdict(Counter)

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if not packet.states:
            raise ValueError("StateFusionPreprocessor requires prior state-producing adapters")

        context = dict(packet.context or {})
        history_window = context.get("history_window")
        if not isinstance(history_window, tuple):
            raise ValueError("StateFusionPreprocessor requires history_window in packet.context")

        numeric_state = next((state for state in reversed(packet.states) if isinstance(state.value, Real)), None)
        if numeric_state is None:
            raise ValueError("StateFusionPreprocessor could not find a numeric state to fuse")

        current = float(numeric_state.value)
        delta = float(context.get("delta", 0.0))
        fused_value = (current, delta, *tuple(float(item) for item in history_window if isinstance(item, Real)))
        context.update(
            {
                "domain": "fusion",
                "value_kind": "fused_tuple",
                "fusion_width": len(fused_value),
            }
        )
        packet.context = context
        packet.states.append(State(value=fused_value, context=context))
        packet.log(self.name, {"fused_value": fused_value, "history": list(history_window)}, role="adapter")
        return packet

    def preprocess(self, raw: Any, *, context: dict[str, Any] | None = None) -> PreprocessedInput:
        packet = AdapterPacket(raw=raw, context=dict(context or {}))
        packet = self.run(packet)
        state = packet.states[-1]
        return PreprocessedInput(state=state, context=dict(state.context), raw=raw, packet=packet)

    def record_transition(self, history_window: Any, next_value: Any) -> None:
        key = _to_float_tuple(history_window)
        if key is None:
            return
        self.transition_memory[key][next_value] += 1

    def predict_transition(self, history_window: Any) -> Any | None:
        key = _to_float_tuple(history_window)
        if key is None:
            return None
        counter = self.transition_memory.get(key)
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def prediction_confidence(self, history_window: Any) -> float:
        """Fraction of observations that agree with the top prediction (0 if unseen)."""
        key = _to_float_tuple(history_window)
        if key is None:
            return 0.0
        counter = self.transition_memory.get(key)
        if not counter:
            return 0.0
        top_count = counter.most_common(1)[0][1]
        return top_count / counter.total()
