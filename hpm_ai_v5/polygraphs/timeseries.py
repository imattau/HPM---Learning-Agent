from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from ..core import State
from .base import PolygraphView, PolygraphGenerator

@dataclass(slots=True)
class TimeSeriesPolygraphGenerator(PolygraphGenerator):
    """Generate multiple views from a sliding window of a time series."""
    window_size: int = 10
    history: list[float] = field(default_factory=list)

    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        if not isinstance(raw, (int, float, np.number)):
            raise TypeError("TimeSeriesPolygraphGenerator expects a numeric value")
        self.history.append(float(raw))
        if len(self.history) > self.window_size:
            self.history.pop(0)
        context = dict(context or {})
        context.setdefault("domain", "timeseries")
        views = []
        
        # 1. Raw value
        views.append(PolygraphView(
            name="raw",
            state=State(value=float(raw), context={**context, "view": "raw"}),
            context={**context, "view": "raw"},
        ))
        
        # 2. Rolling mean (if enough history)
        if len(self.history) >= 2:
            rolling_mean = np.mean(self.history)
            views.append(PolygraphView(
                name="rolling_mean",
                state=State(value=float(rolling_mean), context={**context, "view": "mean"}),
                context={**context, "view": "mean"},
            ))
            
        # 3. Difference (first derivative)
        if len(self.history) >= 2:
            diff = self.history[-1] - self.history[-2]
            views.append(PolygraphView(
                name="diff",
                state=State(value=float(diff), context={**context, "view": "diff"}),
                context={**context, "view": "diff"},
            ))
        return views
