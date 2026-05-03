"""Physics-domain adapters for v5."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..core import State
from .packet import AdapterPacket


@dataclass(slots=True)
class CartpoleStateAdapter:
    """Preprocessor to convert cartpole dict observations to flattened sin/cos tuples
    with action history and derived error metrics."""

    name: str = "cartpole_state"
    action_history_len: int = 3
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["state"])
    _action_buffer: deque[float] = field(init=False)

    def __post_init__(self):
        self._action_buffer = deque([0.0] * self.action_history_len, maxlen=self.action_history_len)

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw = packet.raw
        if not isinstance(raw, dict):
            raise TypeError(f"CartpoleStateAdapter expects a dict, got {type(raw).__name__}")

        pos = float(raw.get("position", 0.0))
        vel = float(raw.get("velocity", 0.0))
        angle = float(raw.get("angle", 0.0))
        ang_vel = float(raw.get("angular_velocity", 0.0))
        
        # Include last action in state if available in context, and update buffer
        last_action = float(packet.context.get("last_action", 0.0))
        self._action_buffer.append(last_action)
        
        # 1. Action history
        actions = tuple(self._action_buffer)

        # 2. Derived stability error: angle^2 + 0.1 * ang_vel^2
        derived_error = angle**2 + 0.1 * ang_vel**2

        # Flattened state: (actions..., pos, vel, sin, cos, ang_vel, derived_error)
        state_value = actions + (pos, vel, math.sin(angle), math.cos(angle), ang_vel, derived_error)

        context = dict(packet.context or {})
        context.update({
            "domain": "physics",
            "task": "cartpole",
            "raw_observation": raw,
            "flattened_state": state_value,
            "action_history": actions,
            "derived_error": derived_error,
        })
        packet.context = context
        packet.states.append(State(value=state_value, context=context))
        packet.log(self.name, {"state": state_value, "derived_error": derived_error}, role="adapter")
        return packet

    def reset(self):
        self._action_buffer = deque([0.0] * self.action_history_len, maxlen=self.action_history_len)


class RunningNormaliserAdapter:
    """Incremental normalisation using sklearn's StandardScaler."""

    name: str = "running_normaliser"

    def __init__(self) -> None:
        self.requires: list[str] = ["state"]
        self.provides: list[str] = ["state"]
        self.scaler = StandardScaler()
        self._fitted = False

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if not packet.states:
            raise ValueError("RunningNormaliserAdapter requires an existing state to normalise")

        last_state = packet.states[-1]
        value = last_state.value

        if isinstance(value, (int, float)):
            vec = np.array([[float(value)]])
        elif isinstance(value, (tuple, list, np.ndarray)):
            vec = np.array([value], dtype=float)
        else:
            raise TypeError(f"RunningNormaliserAdapter cannot normalise {type(value).__name__}")

        # Incremental fit
        self.scaler.partial_fit(vec)
        self._fitted = True

        # Transform
        normalised_vec = self.scaler.transform(vec)[0]
        normalised_value = tuple(normalised_vec) if len(normalised_vec) > 1 else float(normalised_vec[0])

        context = dict(packet.context or {})
        context.update({
            "normalisation_mean": self.scaler.mean_.tolist(),
            "normalisation_scale": self.scaler.scale_.tolist(),
            "normalised_value": normalised_value,
        })
        packet.context = context
        packet.states.append(State(value=normalised_value, context=context))
        packet.log(self.name, {"normalised": normalised_value}, role="adapter")
        return packet

    def reset(self) -> None:
        """Clear normalisation statistics."""
        self.scaler = StandardScaler()
        self._fitted = False


@dataclass(slots=True)
class RewardToGoalAdapter:
    """Convert episodic rewards into a cumulative utility goal for pattern selection."""

    name: str = "reward_to_goal"
    decay: float = 0.9
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=list) # Modifies packet.goal
    _running_utility: float = 0.0

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        # Reward is expected in context (e.g., from environment step)
        reward = float(packet.context.get("reward", 0.0))
        self._running_utility = reward + self.decay * self._running_utility

        if packet.goal is None:
            packet.goal = {}
        
        packet.goal["utility"] = self._running_utility
        packet.log(self.name, {"reward": reward, "running_utility": self._running_utility}, role="adapter")
        return packet

    def reset(self) -> None:
        """Reset running utility at the start of an episode."""
        self._running_utility = 0.0


@dataclass(slots=True)
class TDErrorAdapter:
    """Preprocessor that penalises the utility goal based on forecast error."""

    name: str = "td_error"
    error_alpha: float = 5.0
    requires: list[str] = field(default_factory=lambda: ["state"])
    provides: list[str] = field(default_factory=list)

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        # 1. Get current state (should already be pushed by state adapter)
        if not packet.states:
            return packet
        
        current_state = packet.states[-1].value
        if not isinstance(current_state, (tuple, list, np.ndarray)):
            return packet

        # 2. Get previous forecast from context
        prev_forecast = packet.context.get("prev_forecast")
        if prev_forecast is None:
            return packet
        
        # 3. Compute squared error
        actual = np.array(current_state, dtype=float)
        forecast = np.array(prev_forecast, dtype=float)
        
        # Ensure dimensions match
        if actual.shape != forecast.shape:
            return packet
            
        sq_error = float(np.mean((actual - forecast)**2))
        penalty = -self.error_alpha * sq_error
        
        # 4. Inject into goal utility
        if packet.goal is None:
            packet.goal = {}
        
        current_utility = float(packet.goal.get("utility", 0.0))
        packet.goal["utility"] = current_utility + penalty
        
        packet.log(self.name, {"sq_error": sq_error, "penalty": penalty}, role="adapter")
        return packet
