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
    ) -> None:
        self.action_index = action_index
        self.confidence_threshold = confidence_threshold
        self.max_blend_alpha = max_blend_alpha

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if not isinstance(packet.core_action, Action):
            raise TypeError("CartpoleForecastPostprocessor expects a core Action in packet.core_action")
        packet.validated_output = self.postprocess(packet.core_action, context=packet.context or {})
        packet.log(self.name, {"validated_output": packet.validated_output}, role="adapter")
        return packet

    def postprocess(self, action: Action, *, context: dict[str, Any] | None = None) -> float:
        context = context or {}

        # Always compute the PD heuristic as the stable baseline.
        # Includes cart position/velocity to handle the position boundary (±2.4m)
        # as well as the pole angle limit.
        raw_obs = context.get("raw_observation", {})
        angle = float(raw_obs.get("angle", context.get("angle", 0.0)))
        ang_vel = float(raw_obs.get("angular_velocity", context.get("angular_velocity", 0.0)))
        position = float(raw_obs.get("position", context.get("position", 0.0)))
        velocity = float(raw_obs.get("velocity", context.get("velocity", 0.0)))
        signal = angle + 0.3 * ang_vel + 0.05 * position + 0.02 * velocity
        heuristic_action = 1.0 if signal > 0 else -1.0

        # Attempt to extract and de-normalise the engine's forecast action.
        engine_action: float | None = None
        if (
            action.action_type == "apply_delta"
            and action.forecast is not None
            and action.forecast.value is not None
        ):
            fv = action.forecast.value
            if isinstance(fv, (tuple, list, np.ndarray)) and len(fv) > self.action_index:
                raw_val = float(fv[self.action_index])
                # De-normalise: RunningNormaliserAdapter stores mean/scale in context
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
            result = 1.0 if blended > 0 else -1.0
        else:
            result = heuristic_action

        return result
