"""Numeric postprocessing for the minimal v5 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np

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


class MultiNumericPostprocessor:
    """Postprocessor that clips numeric actions using numpy."""

    name: str = "multi_numeric"
    requires: list[str] = []
    provides: list[str] = ["validated_output"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if not isinstance(packet.core_action, Action):
            raise TypeError("MultiNumericPostprocessor expects a core Action in packet.core_action")
        packet.validated_output = self.postprocess(packet.core_action, context=packet.context or {})
        packet.log(self.name, {"validated_output": packet.validated_output}, role="adapter")
        return packet

    def postprocess(self, action: Action, *, context: dict[str, Any] | None = None) -> float:
        context = context or {}
        
        value = action.value
        # If the core returns a tuple (the forecasted next state), extract the predicted action
        if isinstance(value, (tuple, list, np.ndarray)) and len(value) > 0:
            # Default to the index specified in context, or index 0 if not provided
            idx = int(context.get("action_index", 0))
            if idx < len(value):
                value = value[idx]
            else:
                value = value[0]

        if not isinstance(value, Real):
            raise TypeError(f"MultiNumericPostprocessor expects a real numeric action value, got {type(value).__name__}")

        value = float(value)
        # Default to [-1, 1] for cartpole/continuous control if not specified
        minimum = float(context.get("minimum", -1.0))
        maximum = float(context.get("maximum", 1.0))

        return float(np.clip(value, minimum, maximum))


@dataclass(slots=True)
class ExplorationPostprocessor:
    """Postprocessor that adds decaying Gaussian noise to numeric actions."""

    name: str = "exploration"
    epsilon: float = 0.2
    noise_scale: float = 0.1
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["action"])

    def postprocess(self, action: Action, *, context: dict[str, Any] | None = None) -> float:
        context = context or {}
        value = action.value
        
        # Extract scalar value if it's a tuple (forecast)
        if isinstance(value, (tuple, list, np.ndarray)) and len(value) > 0:
            value = value[0]
            
        if not isinstance(value, (int, float)):
            return 0.0
            
        value = float(value)
        
        # Add noise with probability epsilon
        if np.random.random() < self.epsilon:
            noise = np.random.normal(0, self.noise_scale)
            value += noise
            
        return value
