"""Polygraph generation for physics domains."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..adapter import AdapterPacket
from ..core import State
from .base import PolygraphGenerator, PolygraphView


@dataclass(slots=True)
class PhysicsPolygraphGenerator(PolygraphGenerator):
    """Generates specialized views for physics control tasks."""

    name: str = "physics_polygraph"

    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        context = context or {}
        # The flattened state is already in context from the state adapter
        # state_tuple: (actions..., pos, vel, sin, cos, ang_vel, derived_error)
        state_tuple = context.get("flattened_state")
        if not state_tuple or not isinstance(state_tuple, tuple):
            return []

        # actions are the first 3 elements (for history_len=3)
        actions = state_tuple[:3]
        
        # prev_forecast is also in context
        prev_forecast = context.get("prev_forecast")
        
        views = []
        
        # 1. Raw View (Full state)
        views.append(PolygraphView(
            name="raw_view",
            state=State(value=state_tuple, context={**context, "view": "raw"})
        ))
        
        # Extract variables for specialized views
        # state_tuple: (a1, a2, a3, pos, vel, sin_th, cos_th, th_dot, derived_err)
        sin_th = state_tuple[5]
        cos_th = state_tuple[6]
        th_dot = state_tuple[7]
        th = math.atan2(sin_th, cos_th)
        
        # 2. Stability Error View: (actions..., theta^2, theta_dot^2, theta^2 + 0.1 * theta_dot^2)
        th_sq = th**2
        th_dot_sq = th_dot**2
        stab_err = th_sq + 0.1 * th_dot_sq
        views.append(PolygraphView(
            name="stability_error_view",
            state=State(value=actions + (th_sq, th_dot_sq, stab_err), context={**context, "view": "stability_error"})
        ))

        # 3. Energy View: (actions..., E_kinetic + E_potential)
        # Simplified: KE = 0.5 * (vel^2 + th_dot^2), PE = 1 - cos_th
        vel = state_tuple[4]
        ke = 0.5 * (vel**2 + th_dot**2)
        pe = 1.0 - cos_th
        energy = ke + pe
        views.append(PolygraphView(
            name="energy_view",
            state=State(value=actions + (energy,), context={**context, "view": "energy"})
        ))

        # 4. Prediction Error View: (actions..., MSE between forecast and actual)
        if prev_forecast is not None and len(prev_forecast) == len(state_tuple):
            mse = float(np.mean((np.array(state_tuple) - np.array(prev_forecast))**2))
            views.append(PolygraphView(
                name="prediction_error_view",
                state=State(value=actions + (mse,), context={**context, "view": "prediction_error"})
            ))

        return views
