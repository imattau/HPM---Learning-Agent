"""Adapter-side unpacker for macro actions."""

from __future__ import annotations

from typing import Any

from .packet import AdapterPacket
from ..core import Action


class ActionSequenceUnpacker:
    """Unpack execute-sequence actions into a concrete tuple of steps."""

    name: str = "action_sequence_unpacker"
    requires: list[str] = []
    provides: list[str] = ["validated_output"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if not isinstance(packet.core_action, Action):
            raise TypeError("ActionSequenceUnpacker expects a core Action in packet.core_action")
        packet.validated_output = self.postprocess(packet.core_action, context=packet.context or {})
        packet.log(self.name, {"validated_output": packet.validated_output}, role="adapter")
        return packet

    def postprocess(self, action: Action, *, context: dict[str, Any] | None = None) -> tuple[Any, ...]:
        if action.action_type == "execute_sequence":
            if isinstance(action.value, (list, tuple)):
                return tuple(action.value)
            return (action.value,)
        return (action.value,)

