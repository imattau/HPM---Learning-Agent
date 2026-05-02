"""NumericAdapter — wraps a raw numeric value into a State on the packet."""

from __future__ import annotations

from ..core.state import State
from .packet import AdapterPacket


class NumericAdapter:
    """Append the raw value as a numeric State."""

    name = "numeric"
    requires: list[str] = []
    provides: list[str] = ["state"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        value = float(packet.raw) if packet.raw is not None else 0.0
        packet.states.append(State(value=value))
        packet.log(self.name, {"value": value})
        return packet
