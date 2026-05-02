"""Prefix buffer preprocessing for short-term memory."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from numbers import Real
from typing import Any

from ..adapter import AdapterPacket
from ..core import State
from .base import PreprocessedInput


class PrefixBufferPreprocessor:
    """Attach a fixed-length history buffer to the current observation."""

    name: str = "prefix_buffer"
    requires: list[str] = []
    provides: list[str] = ["state"]

    def __init__(
        self,
        buffer_size: int = 2,
        default: float = -1.0,
        *,
        value_mode: str = "context",
        code_base: int = 10,
        include_history_context: bool = True,
    ) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")
        self.buffer_size = buffer_size
        self.default = float(default)
        self.value_mode = value_mode
        self.code_base = code_base
        self.include_history_context = include_history_context
        self.history: deque[float] = deque(maxlen=buffer_size)
        self._prefix_ids: dict[tuple[float, ...], int] = {}
        self.transition_memory: defaultdict[tuple[float, ...], Counter] = defaultdict(Counter)

    def _encode_code(self, values: list[float]) -> float:
        code = 0
        for value in reversed(values):
            code = code * self.code_base + int(round(value))
        return float(code)

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw = packet.raw
        if not isinstance(raw, Real):
            raise TypeError(f"PrefixBufferPreprocessor expects a real number, got {type(raw).__name__}")

        value = float(raw)
        self.history.append(value)

        padded = [self.default] * (self.buffer_size - len(self.history)) + list(self.history)
        prefix_key = tuple(padded)
        prefix_id = self._prefix_ids.setdefault(prefix_key, len(self._prefix_ids) + 1)
        if self.value_mode == "tuple":
            enriched_value: Any = tuple(padded)
        elif self.value_mode == "code":
            enriched_value = self._encode_code(padded)
        else:
            enriched_value = value
        context = dict(packet.context or {})
        context.update(
            {
                "domain": "prefix_buffer",
                "value_kind": "sequence",
                "buffer_size": self.buffer_size,
                "value_mode": self.value_mode,
                "code_base": self.code_base,
                "prefix_id": prefix_id,
            }
        )
        if self.include_history_context:
            context.update(
                {
                    "raw_history": list(self.history),
                    "history_window": tuple(padded),
                }
            )
        packet.context = context
        packet.states.append(State(value=enriched_value, context=context))
        packet.log(self.name, {"enriched_value": enriched_value, "history": list(self.history)}, role="adapter")
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


def _to_float_tuple(window: Any) -> tuple[float, ...] | None:
    if isinstance(window, (tuple, list)):
        key = tuple(float(item) for item in window if isinstance(item, Real))
        return key if key else None
    return None


# Backward-compatible alias for adapter-centric language.
PrefixBufferAdapter = PrefixBufferPreprocessor
