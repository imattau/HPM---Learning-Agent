"""Delta buffer adapter for short-term trend memory."""

from __future__ import annotations

from numbers import Real
from typing import Any

from .packet import AdapterPacket
from ..core import State


class DeltaBufferAdapter:
    """Attach a fixed-length history of recent deltas to the state."""

    name: str = "delta_buffer"
    requires: list[str] = []
    provides: list[str] = ["state"]

    def __init__(self, buffer_size: int = 2, default: float = 0.0) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")
        self.buffer_size = buffer_size
        self.default = float(default)
        self.history: list[float] = []
        self.previous_value: float | None = None

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw = packet.raw
        if not isinstance(raw, Real):
            raise TypeError(f"DeltaBufferAdapter expects a real number, got {type(raw).__name__}")

        value = float(raw)
        delta = 0.0 if self.previous_value is None else value - self.previous_value
        self.previous_value = value

        self.history.append(delta)
        if len(self.history) > self.buffer_size:
            self.history.pop(0)

        padded = [self.default] * (self.buffer_size - len(self.history)) + list(self.history)
        state_value = tuple(padded)
        context = dict(packet.context or {})
        context.update(
            {
                "domain": "delta_buffer",
                "value_kind": "delta_buffer_tuple",
                "buffer_size": self.buffer_size,
                "delta": delta,
                "delta_window": tuple(padded),
            }
        )
        packet.context = context
        packet.states.append(State(value=state_value, context=context))
        packet.log(self.name, {"delta": delta, "state_value": state_value}, role="adapter")
        return packet
