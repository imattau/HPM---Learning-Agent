"""Trajectory buffer adapter — accumulates sliding window of error/action history."""

from __future__ import annotations

from typing import Any

from .packet import AdapterPacket


class TrajectoryBufferAdapter:
    """Accumulate a sliding window of derived_error and action values via carry_context.

    This adapter enables polygraph views that operate on multi-step trajectories
    rather than single-step snapshots. The window is threaded forward through
    carry_context keys so it survives the preprocessing → engine → postprocessor
    → next-step preprocessing cycle.

    Writes to context:
        error_history       — list of last `window` derived_error values
        action_history_traj — list of last `window` last_action values
        error_trend         — error_history[-1] - error_history[0] (positive = worsening)
        oscillation         — fraction of sign-changes in error delta sequence (0..1)

    Also writes carry keys (carry_error_history, carry_action_history) so the
    next step can restore the window.
    """

    name: str = "trajectory_buffer"
    requires: list[str] = ["state"]
    provides: list[str] = ["state"]

    def __init__(self, window: int = 5) -> None:
        self.window = window

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        ctx = dict(packet.context or {})

        derived_error = float(ctx.get("derived_error", 0.0))
        last_action = float(ctx.get("last_action", 0.0))

        # Restore history from carry_context (pipeline strips "carry_" prefix on delivery;
        # values are tuples to keep them hashable when they enter State contexts)
        error_hist: tuple[float, ...] = tuple(ctx.get("error_history", ()))
        action_hist: tuple[float, ...] = tuple(ctx.get("action_history_traj", ()))

        error_hist = tuple((list(error_hist) + [derived_error])[-self.window:])
        action_hist = tuple((list(action_hist) + [last_action])[-self.window:])

        # Summary features from the window
        if len(error_hist) >= 2:
            error_trend = error_hist[-1] - error_hist[0]
            deltas = tuple(error_hist[i + 1] - error_hist[i] for i in range(len(error_hist) - 1))
            sign_changes = sum(
                1 for i in range(len(deltas) - 1) if deltas[i] * deltas[i + 1] < 0
            )
            oscillation = sign_changes / max(1, len(deltas) - 1)
        else:
            error_trend = 0.0
            oscillation = 0.0

        ctx["error_history"] = error_hist
        ctx["action_history_traj"] = action_hist
        ctx["error_trend"] = error_trend
        ctx["oscillation"] = oscillation
        # Carry forward for next step (pipeline strips carry_ prefix on delivery)
        ctx["carry_error_history"] = error_hist
        ctx["carry_action_history"] = action_hist

        packet.context = ctx
        packet.log(
            self.name,
            {"error_trend": error_trend, "oscillation": oscillation, "window_len": len(error_hist)},
            role="adapter",
        )
        return packet
