"""ARC polygraph views."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from ...adapter import AdapterPacket
from ..common import connected_components, infer_transformation, most_common_colour, normalize_grid
from ..features import colour_delta, grid_delta, object_delta, relation_graph, shape_definitions, structural_delta


def _arc(packet: AdapterPacket) -> dict[str, Any]:
    return packet.context.setdefault("arc", {})


def _flatten_grid(grid):
    return tuple(cell for row in grid for cell in row)


def _edge_map(grid):
    arr = np.asarray(grid, dtype=float)
    gradient = np.hypot(ndi.sobel(arr, axis=0, mode="nearest"), ndi.sobel(arr, axis=1, mode="nearest"))
    return tuple(tuple(int(value > 0.0) for value in row) for row in gradient)


def _distance_transform(grid):
    arr = np.asarray(grid, dtype=int)
    background = most_common_colour(grid)
    mask = arr == background
    distances = ndi.distance_transform_edt(mask)
    return tuple(tuple(int(value) for value in row) for row in distances)


@dataclass
class ImagePolygraph:
    name: str = "image_polygraph"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_examples"]
        self.provides = ["image_polygraph"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task = _arc(packet)["task"]
        for index, example in enumerate(task.train):
            grid = normalize_grid(example.input_grid)
            packet.views.append(
                {
                    "name": self.name,
                    "example_index": index,
                    "flat": _flatten_grid(grid),
                    "edge_map": _edge_map(grid),
                    "distance": _distance_transform(grid),
                }
            )
        packet.log(self.name, {"views": len(packet.views)}, role="adapter")
        return packet


@dataclass
class PixelPolygraph(ImagePolygraph):
    name: str = "pixel_polygraph"

    def __post_init__(self) -> None:
        self.requires = ["arc_examples"]
        self.provides = ["pixel_polygraph"]


@dataclass
class ObjectPolygraph:
    name: str = "object_polygraph"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["image_polygraph"]
        self.provides = ["object_polygraph"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task = _arc(packet)["task"]
        for index, example in enumerate(task.train):
            shapes_4 = shape_definitions(example.input_grid, connectivity=4)
            shapes_8 = shape_definitions(example.input_grid, connectivity=8)
            graph_4 = relation_graph(shapes_4)
            graph_8 = relation_graph(shapes_8)
            packet.views.append(
                {
                    "name": self.name,
                    "example_index": index,
                    "objects_4": [asdict(obj) for obj in connected_components(example.input_grid, connectivity=4)],
                    "objects_8": [asdict(obj) for obj in connected_components(example.input_grid, connectivity=8)],
                    "relation_graph_4": graph_4,
                    "relation_graph_8": graph_8,
                    "relation_graph_4_summary": {
                        "nodes": len(shapes_4),
                        "edges": graph_4.number_of_edges(),
                    },
                    "relation_graph_8_summary": {
                        "nodes": len(shapes_8),
                        "edges": graph_8.number_of_edges(),
                    },
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
        self.requires = ["transformation_polygraph"]
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


@dataclass
class TransformationPolygraph:
    name: str = "transformation_polygraph"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["object_polygraph"]
        self.provides = ["transformation_polygraph"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task = _arc(packet)["task"]
        for index, example in enumerate(task.train):
            if example.output_grid is None:
                continue
            packet.views.append(
                {
                    "name": self.name,
                    "example_index": index,
                    "grid_delta": grid_delta(example.input_grid, example.output_grid),
                    "object_delta": object_delta(example.input_grid, example.output_grid),
                    "colour_delta": colour_delta(example.input_grid, example.output_grid),
                    "structural_delta": structural_delta(example.input_grid, example.output_grid),
                }
            )
        packet.log(self.name, {"views": len(packet.views)}, role="adapter")
        return packet
