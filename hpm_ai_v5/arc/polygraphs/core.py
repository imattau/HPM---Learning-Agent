"""ARC polygraph views."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ...adapter import AdapterPacket
from ..common import connected_components, infer_transformation, normalize_grid


def _arc(packet: AdapterPacket) -> dict[str, Any]:
    return packet.context.setdefault("arc", {})


@dataclass
class PixelPolygraph:
    name: str = "pixel_polygraph"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_examples"]
        self.provides = ["pixel_polygraph"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task = _arc(packet)["task"]
        for index, example in enumerate(task.train):
            if example.output_grid is None:
                continue
            input_grid = normalize_grid(example.input_grid)
            output_grid = normalize_grid(example.output_grid)
            if len(input_grid) == len(output_grid) and len(input_grid[0]) == len(output_grid[0]):
                diff = [
                    [1 if input_grid[row][col] != output_grid[row][col] else 0 for col in range(len(input_grid[0]))]
                    for row in range(len(input_grid))
                ]
                fragmentation = sum(sum(row) for row in diff)
            else:
                diff = None
                fragmentation = abs(len(input_grid) - len(output_grid)) + abs(len(input_grid[0]) - len(output_grid[0]))
            packet.views.append({"name": self.name, "example_index": index, "diff": diff, "fragmentation": fragmentation})
        packet.log(self.name, {"views": len(packet.views)}, role="adapter")
        return packet


@dataclass
class ObjectPolygraph:
    name: str = "object_polygraph"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["pixel_polygraph"]
        self.provides = ["object_polygraph"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task = _arc(packet)["task"]
        for index, example in enumerate(task.train):
            packet.views.append(
                {
                    "name": self.name,
                    "example_index": index,
                    "objects": [asdict(obj) for obj in connected_components(example.input_grid)],
                }
            )
        packet.log(self.name, {"views": len(packet.views)}, role="adapter")
        return packet


@dataclass
class ColourPolygraph:
    name: str = "colour_polygraph"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["object_polygraph"]
        self.provides = ["colour_polygraph"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task = _arc(packet)["task"]
        for index, example in enumerate(task.train):
            if example.output_grid is None:
                continue
            candidate = infer_transformation(example.input_grid, example.output_grid)
            if candidate is None:
                continue
            packet.views.append(
                {
                    "name": self.name,
                    "example_index": index,
                    "colour_map": dict(candidate.colour_map),
                    "score": 1.0,
                }
            )
        packet.log(self.name, {"views": len(packet.views)}, role="adapter")
        return packet


@dataclass
class GeometryPolygraph:
    name: str = "geometry_polygraph"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["colour_polygraph"]
        self.provides = ["geometry_polygraph"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task = _arc(packet)["task"]
        for index, example in enumerate(task.train):
            if example.output_grid is None:
                continue
            candidate = infer_transformation(example.input_grid, example.output_grid)
            if candidate is None:
                continue
            packet.views.append(
                {
                    "name": self.name,
                    "example_index": index,
                    "dx": candidate.dx,
                    "dy": candidate.dy,
                    "score": 1.0 / (1.0 + abs(candidate.dx) + abs(candidate.dy)),
                }
            )
        packet.log(self.name, {"views": len(packet.views)}, role="adapter")
        return packet
