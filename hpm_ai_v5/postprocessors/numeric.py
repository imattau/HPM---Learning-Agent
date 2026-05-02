"""Numeric postprocessing for the minimal v5 pipeline."""

from __future__ import annotations

from numbers import Real
from typing import Any

from ..adapter import AdapterPacket
from ..core import Action
from .base import PostprocessedOutput


class NumericPostprocessor:
    """Validate and render numeric actions."""

    name: str = "numeric"
    requires: list[str] = []
    provides: list[str] = ["validated_output"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if not isinstance(packet.core_action, Action):
            raise TypeError("NumericPostprocessor expects a core Action in packet.core_action")
        packet.validated_output = self.postprocess(packet.core_action, context=packet.context or {})
        packet.log(self.name, {"validated_output": packet.validated_output}, role="adapter")
        return packet

    def postprocess(self, action: Action, *, context: dict[str, Any] | None = None) -> float:
        context = context or {}
        if action.action_type != "apply_delta":
            raise ValueError(f"Cannot postprocess action_type={action.action_type!r}")
        if not isinstance(action.value, Real):
            raise TypeError("NumericPostprocessor expects a real numeric action value")

        value = float(action.value)
        minimum = context.get("minimum")
        maximum = context.get("maximum")
        if minimum is not None and value < float(minimum):
            raise ValueError("Rendered value is below the domain minimum")
        if maximum is not None and value > float(maximum):
            raise ValueError("Rendered value exceeds the domain maximum")
        return value
