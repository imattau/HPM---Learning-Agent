"""Validation-only adapter."""

from __future__ import annotations

from typing import Any

from .packet import AdapterPacket
from ..core import Action


class ValidationOnlyAdapter:
    """Validate that the action exists and pass it through unchanged."""

    name: str = "validation_only"
    requires: list[str] = []
    provides: list[str] = ["validated_output"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if not isinstance(packet.core_action, Action):
            raise TypeError("ValidationOnlyAdapter expects a core Action in packet.core_action")
        packet.validated_output = self.postprocess(packet.core_action, context=packet.context or {})
        packet.log(self.name, {"validated_output": packet.validated_output}, role="adapter")
        return packet

    def postprocess(self, action: Action, *, context: dict[str, Any] | None = None) -> Any:
        if action is None:
            raise ValueError("ValidationOnlyAdapter requires a concrete action")
        return action.value
