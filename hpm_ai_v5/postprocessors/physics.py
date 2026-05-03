"""Physics-domain postprocessors for v5."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..adapter import AdapterPacket
from ..core import Action


class CartpoleForecastPostprocessor:
    """Convert a PatternEngine forecast into a CartPole action.

    The engine forecasts the next state tuple:
        (a0, a1, ..., a_{N-1}, pos, vel, sin_theta, cos_theta, ang_vel, error)

    The most recent action in that next state's history buffer (index N-1) is
    the action predicted to have been taken to produce that state — i.e. the
    action to execute now.

    The forecast is in normalised space (StandardScaler). We de-normalise using
    the scaler statistics stored in packet.context by RunningNormaliserAdapter.

    A PD heuristic is always computed as the baseline. The engine action is
    blended in only when engine confidence >= confidence_threshold, preventing
    chaotic overrides during early sparse-pattern episodes.

    Bang-bang output (±1) is used because CartPole's threshold-based termination
    rewards decisive correction over proportional control near balance.
    """

    name: str = "cartpole_forecast"
    requires: list[str] = []
    provides: list[str] = ["validated_output"]

    def __init__(
        self,
        action_index: int = 2,
        confidence_threshold: float = 0.6,
        max_blend_alpha: float = 0.4,
        q_alpha: float = 0.15,
        q_gamma: float = 0.9,
        q_epsilon_start: float = 0.3,
    ) -> None:
        self.action_index = action_index
        self.confidence_threshold = confidence_threshold
        self.max_blend_alpha = max_blend_alpha
        self.q_alpha = q_alpha
        self.q_gamma = q_gamma
        self.q_epsilon = q_epsilon_start
        self.q_table: dict[tuple, float] = {}
        self._prev_state_key: tuple | None = None
        self._prev_action: float = 0.0

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if not isinstance(packet.core_action, Action):
            raise TypeError("CartpoleForecastPostprocessor expects a core Action in packet.core_action")
        packet.validated_output = self.postprocess(packet.core_action, context=packet.context or {})
        packet.log(self.name, {"validated_output": packet.validated_output}, role="adapter")
        return packet

    def postprocess(self, action: Action, *, context: dict[str, Any] | None = None) -> float:
        context = context or {}

        # Weak baseline: sign(angle) only — ~40 steps alone.
        raw_obs = context.get("raw_observation", {})
        angle = float(raw_obs.get("angle", context.get("angle", 0.0)))
        heuristic_action = 1.0 if angle > 0 else -1.0

        # Attempt to extract and de-normalise the engine's forecast action.
        engine_action: float | None = None
        if (
            action.action_type == "apply_delta"
            and action.forecast is not None
            and action.forecast.value is not None
        ):
            fv = action.forecast.value
            if isinstance(fv, (tuple, list, np.ndarray)) and len(fv) > 0:
                if len(fv) <= 4:
                    raw_val = float(np.sign(fv[0])) if fv[0] != 0.0 else 1.0
                    engine_action = float(np.clip(raw_val, -1.0, 1.0))
                elif len(fv) > self.action_index:
                    raw_val = float(fv[self.action_index])
                    means = context.get("normalisation_mean", [])
                    scales = context.get("normalisation_scale", [])
                    if (
                        isinstance(means, (list, tuple))
                        and isinstance(scales, (list, tuple))
                        and len(means) > self.action_index
                        and len(scales) > self.action_index
                    ):
                        scale = float(scales[self.action_index])
                        mean = float(means[self.action_index])
                        if scale > 1e-8:
                            raw_val = raw_val * scale + mean
                    engine_action = float(np.clip(raw_val, -1.0, 1.0))

        # Blend engine action in when confidence is sufficient.
        confidence = action.confidence
        if engine_action is not None and confidence >= self.confidence_threshold:
            alpha = min(self.max_blend_alpha, (confidence - self.confidence_threshold) * 2.0)
            blended = (1.0 - alpha) * heuristic_action + alpha * engine_action
            heuristic_result = 1.0 if blended > 0 else -1.0
        else:
            heuristic_result = heuristic_action

        # --- Q-table policy ---
        reward = float(context.get("reward", 0.0))

        # Compute binary sign state key from raw observation
        raw_obs = context.get("raw_observation", {})
        angle = float(raw_obs.get("angle", 0.0))
        ang_vel = float(raw_obs.get("angular_velocity", 0.0))
        sign_angle = 1.0 if angle > 0 else -1.0
        sign_ang_vel = 1.0 if ang_vel > 0 else -1.0
        q_state_key = (sign_angle, sign_ang_vel)

        # Q-update from previous step
        if self._prev_state_key is not None:
            key_prev = (self._prev_state_key, self._prev_action)
            q_prev = self.q_table.get(key_prev, 0.0)
            q_next_pos = self.q_table.get((q_state_key, 1.0), 0.0)
            q_next_neg = self.q_table.get((q_state_key, -1.0), 0.0)
            max_q_next = max(q_next_pos, q_next_neg)
            td_target = reward + self.q_gamma * max_q_next
            self.q_table[key_prev] = q_prev + self.q_alpha * (td_target - q_prev)

        # Q-action selection
        q_pos = self.q_table.get((q_state_key, 1.0), 0.0)
        q_neg = self.q_table.get((q_state_key, -1.0), 0.0)
        if abs(q_pos - q_neg) > 0.01:
            q_action = 1.0 if q_pos > q_neg else -1.0
            q_confidence = abs(q_pos - q_neg) / (abs(q_pos) + abs(q_neg) + 1e-8)
        else:
            q_action = heuristic_result
            q_confidence = 0.0

        # ε-greedy: with probability q_epsilon, explore with random action
        import random
        if self.q_epsilon > 0 and random.random() < self.q_epsilon:
            result = 1.0 if random.random() < 0.5 else -1.0
        else:
            # Greedy: use Q-action if confident, else heuristic
            result = q_action if q_confidence > 0.1 else heuristic_result

        # Write carry_context (prefixed with carry_) into context dict
        context["carry_q_action"] = result
        context["carry_q_confidence"] = q_confidence
        context["carry_q_value"] = max(q_pos, q_neg)
        context["carry_q_state_key"] = q_state_key
        context["carry_q_table_size"] = len(self.q_table)
        context["carry_q_epsilon"] = self.q_epsilon

        # Update state for next step
        self._prev_state_key = q_state_key
        self._prev_action = result

        return result


class BinaryExplorationPostprocessor:
    """Randomly flip the action sign to create exploration diversity.

    Unlike Gaussian noise, binary flipping produces clear differential signals:
    the action polygraph engines see both action=+1 and action=-1 in similar
    states, enabling reward-based discrimination between correct and incorrect
    actions. Epsilon decays across episodes as the engine accumulates policy.
    """

    name: str = "binary_exploration"
    requires: list[str] = ["cartpole_forecast"]
    provides: list[str] = ["validated_output"]

    def __init__(self, epsilon: float = 0.3) -> None:
        self.epsilon = epsilon

    def run(self, packet: "AdapterPacket") -> "AdapterPacket":
        if packet.validated_output is None:
            return packet
        action = float(packet.validated_output)
        if np.random.random() < self.epsilon:
            action = -action
        packet.validated_output = action
        packet.log(self.name, {"action": action, "epsilon": self.epsilon}, role="adapter")
        return packet

    def postprocess(self, action: "Action", *, context: Any | None = None) -> float:
        return float(action.value) if action.value is not None else 0.0
