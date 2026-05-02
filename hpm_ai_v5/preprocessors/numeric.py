"""Numeric preprocessing for the minimal v5 pipeline."""

from __future__ import annotations

from numbers import Real
from typing import Any

from ..adapter import AdapterPacket
from ..core import Delta, State
from .base import PreprocessedInput


class NumericPreprocessor:
    """Convert a numeric stream into state + delta form."""

    name: str = "numeric"
    requires: list[str] = []
    provides: list[str] = ["state"]
    previous_value: float | None = None

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if not isinstance(packet.raw, Real):
            raise TypeError(f"NumericPreprocessor expects a real number, got {type(packet.raw).__name__}")
        value = float(packet.raw)
        packet.clean = value
        context = dict(packet.draft_output or {})
        context.update({"domain": "numeric", "value_kind": "scalar"})
        if self.previous_value is None:
            context["delta"] = 0.0
        else:
            context["delta"] = Delta.between(self.previous_value, value, level="state").value
        packet.states.append(State(value=value, context=context))
        packet.log(self.name, {"context": context})
        self.previous_value = value
        return packet

    def preprocess(self, raw: Any, *, context: dict[str, Any] | None = None) -> PreprocessedInput:
        packet = AdapterPacket(raw=raw)
        if context:
            packet.draft_output = dict(context)
        packet = self.run(packet)
        state = packet.states[-1]
        return PreprocessedInput(state=state, context=dict(state.context), raw=raw, packet=packet)
