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

    A simple heuristic is always computed as the baseline. The engine forecast
    is treated as an auxiliary directional hint, not as a replacement policy.

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
        self.learning_enabled = True
        self.q_table: dict[tuple, float] = {}
        self._prev_state_key: tuple | None = None
        self._prev_action: float = 0.0

    def export_state(self) -> dict[str, Any]:
        return {
            "q_alpha": self.q_alpha,
            "q_gamma": self.q_gamma,
            "q_epsilon": self.q_epsilon,
            "learning_enabled": self.learning_enabled,
            "q_table": dict(self.q_table),
            "prev_state_key": self._prev_state_key,
            "prev_action": self._prev_action,
        }

    def import_state(self, state: dict[str, Any]) -> None:
        self.q_alpha = float(state.get("q_alpha", self.q_alpha))
        self.q_gamma = float(state.get("q_gamma", self.q_gamma))
        self.q_epsilon = float(state.get("q_epsilon", self.q_epsilon))
        self.learning_enabled = bool(state.get("learning_enabled", self.learning_enabled))
        self.q_table = dict(state.get("q_table", {}))
        self._prev_state_key = state.get("prev_state_key")
        self._prev_action = float(state.get("prev_action", 0.0))

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if not isinstance(packet.core_action, Action):
            raise TypeError("CartpoleForecastPostprocessor expects a core Action in packet.core_action")
        packet.validated_output = self.postprocess(packet.core_action, context=packet.context or {})
        packet.log(self.name, {"validated_output": packet.validated_output}, role="adapter")
        return packet

    @staticmethod
    def _fallback_q_state_key(raw_obs: dict[str, Any]) -> tuple[float, float]:
        angle = float(raw_obs.get("angle", 0.0))
        ang_vel = float(raw_obs.get("angular_velocity", 0.0))
        sign_angle = 1.0 if angle > 0 else -1.0
        sign_ang_vel = 1.0 if ang_vel > 0 else -1.0
        return (sign_angle, sign_ang_vel)

    @staticmethod
    def _magnitude_bucket(value: float, thresholds: tuple[float, float]) -> float:
        magnitude = abs(value)
        if magnitude < thresholds[0]:
            bucket = 0.0
        elif magnitude < thresholds[1]:
            bucket = 1.0
        else:
            bucket = 2.0
        if value == 0.0:
            return 0.0
        return bucket if value > 0 else -bucket

    def _policy_state_key(self, action: Action, raw_obs: dict[str, Any]) -> tuple[Any, ...]:
        angle = float(raw_obs.get("angle", 0.0))
        ang_vel = float(raw_obs.get("angular_velocity", 0.0))
        position = float(raw_obs.get("position", 0.0))
        velocity = float(raw_obs.get("velocity", 0.0))
        return (
            "cartpole_q",
            self._magnitude_bucket(angle, (0.03, 0.10)),
            self._magnitude_bucket(ang_vel, (0.25, 0.75)),
            self._magnitude_bucket(position, (0.5, 1.5)),
            self._magnitude_bucket(velocity, (0.25, 0.75)),
        )

    def postprocess(self, action: Action, *, context: dict[str, Any] | None = None) -> float:
        context = context or {}

        # Keep the baseline policy simple and interpretable.
        raw_obs = context.get("raw_observation", {})
        angle = float(raw_obs.get("angle", context.get("angle", 0.0)))
        ang_vel = float(raw_obs.get("angular_velocity", 0.0))
        position = float(raw_obs.get("position", 0.0))
        velocity = float(raw_obs.get("velocity", 0.0))
        signal = angle + (0.3 * ang_vel) + (0.05 * position) + (0.02 * velocity)
        heuristic_action = 1.0 if signal > 0 else -1.0

        # Treat the engine forecast as a weak directional hint only.
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

        confidence = action.confidence
        if engine_action is not None and confidence >= self.confidence_threshold:
            alpha = min(0.2, (confidence - self.confidence_threshold) * 1.0)
            blended = (1.0 - alpha) * heuristic_action + alpha * engine_action
            heuristic_result = 1.0 if blended > 0 else -1.0
        else:
            heuristic_result = heuristic_action

        # --- Delta-error shaped reward ---
        # Negative when error grew (wrong action), positive when it shrank.
        # Terminal steps get a large negative reward (-1.0).
        _MAX_ERROR = 0.15
        derived_error = float(context.get("derived_error", 0.0))
        prev_derived_error = float(context.get("prev_derived_error", 0.0))
        binary_reward = float(context.get("reward", 0.0))

        if binary_reward == 0.0:
            shaped_reward = -1.0
        else:
            delta_error = derived_error - prev_derived_error
            shaped_reward = -delta_error / _MAX_ERROR
            shaped_reward = max(-1.0, min(1.0, shaped_reward))

        raw_obs = context.get("raw_observation", {})
        q_state_key = self._policy_state_key(action, raw_obs)

        # Q-update for primary Q-table from previous step using shaped reward
        if self.learning_enabled and self._prev_state_key is not None:
            key_prev = (self._prev_state_key, self._prev_action)
            q_prev = self.q_table.get(key_prev, 0.0)
            q_next_pos = self.q_table.get((q_state_key, 1.0), 0.0)
            q_next_neg = self.q_table.get((q_state_key, -1.0), 0.0)
            max_q_next = max(q_next_pos, q_next_neg)
            td_target = shaped_reward + self.q_gamma * max_q_next
            self.q_table[key_prev] = q_prev + self.q_alpha * (td_target - q_prev)

        # Q-action selection from primary table
        q_pos = self.q_table.get((q_state_key, 1.0), 0.0)
        q_neg = self.q_table.get((q_state_key, -1.0), 0.0)
        q_diff = abs(q_pos - q_neg)
        if q_diff > 0.01:
            q_action = 1.0 if q_pos > q_neg else -1.0
            q_confidence = q_diff
        else:
            q_action = heuristic_result
            q_confidence = 0.0

        if q_confidence >= 0.05:
            greedy_result = q_action
            selected_source = "primary_q"
            selected_confidence = min(1.0, q_confidence)
        else:
            greedy_result = heuristic_result
            selected_source = "heuristic"
            selected_confidence = 0.0

        # ε-greedy exploration
        import random
        effective_epsilon = self.q_epsilon if self.learning_enabled else 0.0
        if effective_epsilon > 0 and random.random() < effective_epsilon:
            result = 1.0 if random.random() < 0.5 else -1.0
        else:
            result = greedy_result

        # Write carry_context
        context["carry_q_action"] = result
        context["carry_q_confidence"] = q_confidence
        context["carry_q_value"] = max(q_pos, q_neg)
        context["carry_q_state_key"] = q_state_key
        context["carry_q_table_size"] = len(self.q_table)
        context["carry_q_epsilon"] = self.q_epsilon
        context["carry_action_hypothesis_source"] = selected_source
        context["carry_action_hypothesis_confidence"] = selected_confidence
        context["carry_prev_derived_error"] = derived_error  # this step's error becomes next step's prev

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
