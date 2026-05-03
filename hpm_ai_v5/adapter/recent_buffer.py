"""Recent buffer adapter for short-term history."""

from __future__ import annotations

from numbers import Real
from typing import Any

from .packet import AdapterPacket
from ..core import State


class RecentBufferAdapter:
    """Attach a fixed-length history of recent raw values to the state."""

    name: str = "recent_buffer"
    requires: list[str] = []
    provides: list[str] = ["state"]

    def __init__(self, buffer_size: int = 2, default: Any = -1.0) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")
        self.buffer_size = buffer_size
        self.default = default
        self.history: list[Any] = []

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw = packet.raw
        if isinstance(raw, Real):
            value: Any = float(raw)
        else:
            value = raw

        self.history.append(value)
        if len(self.history) > self.buffer_size:
            self.history.pop(0)

        padded = [self.default] * (self.buffer_size - len(self.history)) + list(self.history)
        state_value = tuple(padded)
        context = dict(packet.context or {})
        context.update(
            {
                "domain": "recent_buffer",
                "value_kind": "buffer_tuple",
                "buffer_size": self.buffer_size,
                "raw_history": list(self.history),
                "history_window": tuple(padded),
            }
        )
        packet.context = context
        packet.states.append(State(value=state_value, context=context))
        packet.log(self.name, {"history": list(self.history), "state_value": state_value}, role="adapter")
        return packet
