"""Connected components adapter for ARC-style object extraction."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .arc_utils import ArcObject, connected_components, grid_context, normalize_grid
from .packet import AdapterPacket
from ..core import State


class ConnectedComponentsAdapter:
    """Extract connected components and emit a stable structural signature."""

    name: str = "connected_components"
    requires: list[str] = []
    provides: list[str] = ["state"]

    def __init__(self, background: int | None = None) -> None:
        self.background = background

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        grid = normalize_grid(packet.raw)  # type: ignore[arg-type]
        components = connected_components(grid, background=self.background)
        signatures = tuple(
            (
                component.colour,
                component.size,
                component.bbox[0],
                component.bbox[1],
                component.bbox[2],
                component.bbox[3],
                round(component.centroid[0], 3),
                round(component.centroid[1], 3),
            )
            for component in components
        )
        context = dict(packet.context or {})
        context.update(
            {
                "domain": "grid",
                "value_kind": "components_tuple",
                "component_count": len(components),
                "background": self.background if self.background is not None else grid_context(grid)["background"],
                "components": [asdict(component) for component in components],
                **grid_context(grid),
            }
        )
        packet.context = context
        packet.states.append(State(value=signatures, context=context))
        packet.log(self.name, {"component_count": len(components)}, role="adapter")
        return packet
