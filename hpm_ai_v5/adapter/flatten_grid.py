"""Grid flattening adapter for ARC-style structure."""

from __future__ import annotations

from typing import Any, Sequence

from .arc_utils import grid_context, normalize_grid
from .packet import AdapterPacket
from ..core import State


class FlattenGridAdapter:
    """Flatten a 2D grid into a row-major tuple."""

    name: str = "flatten_grid"
    requires: list[str] = []
    provides: list[str] = ["state"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        grid = normalize_grid(packet.raw)  # type: ignore[arg-type]
        state_value = tuple(cell for row in grid for cell in row)
        context = dict(packet.context or {})
        context.update({"domain": "grid", "value_kind": "flattened_grid", **grid_context(grid)})
        packet.context = context
        packet.states.append(State(value=state_value, context=context))
        packet.log(self.name, {"shape": (len(grid), len(grid[0]) if grid else 0)}, role="adapter")
        return packet
