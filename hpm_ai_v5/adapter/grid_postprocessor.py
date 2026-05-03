"""Grid reconstruction adapter."""

from __future__ import annotations

from typing import Any, Sequence

from .packet import AdapterPacket
from ..core import Action


class GridPostprocessor:
    """Reshape flattened grid outputs into 2D lists."""

    name: str = "grid_postprocessor"
    requires: list[str] = []
    provides: list[str] = ["validated_output"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if not isinstance(packet.core_action, Action):
            raise TypeError("GridPostprocessor expects a core Action in packet.core_action")
        packet.validated_output = self.postprocess(packet.core_action, context=packet.context or {})
        packet.log(self.name, {"validated_output": packet.validated_output}, role="adapter")
        return packet

    def postprocess(self, action: Action, *, context: dict[str, Any] | None = None) -> list[list[Any]]:
        context = context or {}
        if not isinstance(action.value, (tuple, list)):
            raise TypeError("GridPostprocessor expects a flat tuple or list of grid values")
        height = context.get("height")
        width = context.get("width")
        if height is None or width is None:
            raise ValueError("GridPostprocessor requires height and width in context")
        flat = list(action.value)
        if len(flat) != int(height) * int(width):
            raise ValueError("Flat grid length does not match height*width")
        return [flat[index : index + int(width)] for index in range(0, len(flat), int(width))]

