from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import scipy.ndimage as ndi
from skimage import feature, measure
from ..core import State
from .base import PolygraphView, PolygraphGenerator

@dataclass(slots=True)
class GridPolygraphGenerator(PolygraphGenerator):
    """Generate multiple views from a 2D grid of integers."""
    connectivity: int = 1 # for components

    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        if not isinstance(raw, (list, np.ndarray)):
            raise TypeError("GridPolygraphGenerator expects a 2D list or numpy array")
        grid = np.array(raw, dtype=np.int32)
        context = dict(context or {})
        context.setdefault("domain", "grid")
        views = []
        
        # 1. Flattened grid
        flat = tuple(map(int, grid.flatten()))
        views.append(PolygraphView(
            name="flattened",
            state=State(value=flat, context={**context, "view": "flat", "shape": grid.shape}),
            context={**context, "view": "flat"},
        ))
        
        # 2. Edge map (Canny) – output as flattened binary tuple
        edges = feature.canny(grid.astype(float))
        edges_flat = tuple(map(int, edges.flatten().astype(int)))
        views.append(PolygraphView(
            name="edges",
            state=State(value=edges_flat, context={**context, "view": "edges"}),
            context={**context, "view": "edges"},
        ))
        
        # 3. Connected components (labels)
        labels = measure.label(grid, connectivity=self.connectivity)
        comps_flat = tuple(map(int, labels.flatten()))
        views.append(PolygraphView(
            name="components",
            state=State(value=comps_flat, context={**context, "view": "components"}),
            context={**context, "view": "components"},
        ))
        
        # 4. Distance transform (from background)
        binary = (grid != 0).astype(bool)
        dist = ndi.distance_transform_edt(binary)
        dist_flat = tuple(map(float, dist.flatten()))
        views.append(PolygraphView(
            name="distance",
            state=State(value=dist_flat, context={**context, "view": "distance"}),
            context={**context, "view": "distance"},
        ))
        return views
