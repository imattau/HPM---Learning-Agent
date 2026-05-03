"""Action-policy polygraph generator for CartPole control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..core import State
from .base import PolygraphGenerator, PolygraphView


@dataclass(slots=True)
class ActionPolygraphGenerator(PolygraphGenerator):
    """Generates compact views emphasising action-state interactions for control policy learning."""

    name: str = "action_policy_polygraph"

    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        context = context or {}

        flattened_state = context.get("flattened_state")
        if not flattened_state or not isinstance(flattened_state, tuple) or len(flattened_state) < 9:
            return []

        raw_obs = context.get("raw_observation", {})
        angle = float(raw_obs.get("angle", 0.0))
        ang_vel = float(raw_obs.get("angular_velocity", 0.0))
        position = float(raw_obs.get("position", 0.0))

        # Action history (indices 0-2 in the 9-dim state)
        a0 = float(flattened_state[0])
        a1 = float(flattened_state[1])
        a2 = float(flattened_state[2])  # most recent action

        last_action = a2

        views = []

        # View 1 — action_angle
        sign_angle = 1.0 if angle > 0 else -1.0
        sign_ang_vel = 1.0 if ang_vel > 0 else -1.0
        angle_mag = min(1.0, abs(angle) / 0.2095)
        v1 = (last_action, sign_angle, sign_ang_vel, angle_mag)
        if all(math.isfinite(x) for x in v1):
            views.append(PolygraphView(
                name="action_angle",
                state=State(value=v1, context={**context, "view": "action_angle"})
            ))

        # View 2 — action_stability
        stability_error = angle ** 2 + 0.1 * ang_vel ** 2
        v2 = (last_action, stability_error, sign_angle)
        if all(math.isfinite(x) for x in v2):
            views.append(PolygraphView(
                name="action_stability",
                state=State(value=v2, context={**context, "view": "action_stability"})
            ))

        # View 3 — action_alignment
        # last_action at fv[0] so forecast fv[0] = action_now.
        # action_angle_product captures whether the action aligned with the lean.
        action_angle_product = last_action * sign_angle
        ang_vel_clamped = max(-1.0, min(1.0, ang_vel))
        position_sign = 1.0 if position > 0 else -1.0
        v3 = (last_action, action_angle_product, ang_vel_clamped, position_sign)
        if all(math.isfinite(x) for x in v3):
            views.append(PolygraphView(
                name="action_alignment",
                state=State(value=v3, context={**context, "view": "action_alignment"})
            ))

        # View 4 — action_history_pattern
        # Reversed (most recent first) so forecast fv[0] = action_now.
        v4 = (a2, a1, a0)
        if all(math.isfinite(x) for x in v4):
            views.append(PolygraphView(
                name="action_history_pattern",
                state=State(value=v4, context={**context, "view": "action_history_pattern"})
            ))

        # View 5 — binary_sign: 8 states = (sign_angle, sign_ang_vel, sign_action).
        # 4-state key for Q-table (sign_angle, sign_ang_vel) — converges quickly
        # with few episodes. The shaped reward (derived_error-based) provides the
        # gradient signal that distinguishes correct from incorrect actions within
        # each quadrant, compensating for the coarse state resolution.
        sign_action = 1.0 if last_action > 0 else -1.0
        v5 = (sign_angle, sign_ang_vel, sign_action)
        if all(math.isfinite(x) for x in v5):
            views.append(PolygraphView(
                name="binary_sign",
                state=State(value=v5, context={**context, "view": "binary_sign"})
            ))

        return views
