"""Numeric polygraph generation for v5."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

from ..core import Delta, State
from .base import PolygraphView


@dataclass(slots=True)
class NumericPolygraphGenerator:
    """Generate three simple numeric views.

    The views are intentionally small:
    - value_delta: raw numeric delta
    - trend: sign of the delta
    - noisy_delta: delta with an alternating offset
    """

    previous_value: float | None = None
    step: int = 0

    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        if not isinstance(raw, Real):
            raise TypeError(f"NumericPolygraphGenerator expects a real number, got {type(raw).__name__}")

        value = float(raw)
        context = dict(context or {})
        context.setdefault("domain", "numeric")

        if self.previous_value is None:
            delta_value = 0.0
        else:
            delta_value = float(Delta.between(self.previous_value, value, level="state").value)

        trend_value = 0.0
        if delta_value > 0:
            trend_value = 1.0
        elif delta_value < 0:
            trend_value = -1.0

        noise = 0.0 if self.step % 2 == 0 else 1.5

        views = [
            PolygraphView(
                name="value_delta",
                state=State(value=value, context={**context, "view": "value_delta", "delta": delta_value}),
                context={**context, "view": "value_delta"},
            ),
            PolygraphView(
                name="trend",
                state=State(value=trend_value, context={**context, "view": "trend"}),
                context={**context, "view": "trend"},
            ),
            PolygraphView(
                name="noisy_delta",
                state=State(value=value + noise, context={**context, "view": "noisy_delta", "delta": delta_value}),
                context={**context, "view": "noisy_delta"},
            ),
        ]

        self.previous_value = value
        self.step += 1
        return views
