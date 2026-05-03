"""ARC structural adapters for spatial, object, and delta analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ...adapter import AdapterPacket
from ...core import State
from ..common import ArcTask, connected_components, grid_context, normalize_grid
from ..features import (
    color_summary,
    colour_delta,
    edge_list,
    grid_delta,
    object_delta,
    pattern_miner,
    shape_definitions,
    spatial_relations,
    structural_delta,
)


def _arc(packet: AdapterPacket) -> dict[str, Any]:
    return packet.context.setdefault("arc", {})


def _source_grids(task: ArcTask) -> list[tuple[str, int, tuple[tuple[int, ...], ...]]]:
    items: list[tuple[str, int, tuple[tuple[int, ...], ...]]] = []
    for index, example in enumerate(task.train):
        items.append(("train", index, example.input_grid))
    for index, grid in enumerate(task.test_inputs):
        items.append(("test", index, grid))
    return items


def _append_state(packet: AdapterPacket, *, value: Any, grid: tuple[tuple[int, ...], ...], split: str, index: int, value_kind: str, extra: dict[str, Any] | None = None) -> None:
    context = dict(packet.context or {})
    context.update({"domain": "grid", "split": split, "index": index, "value_kind": value_kind, **grid_context(grid)})
    if extra:
        context.update(extra)
    packet.context = context
    packet.states.append(State(value=value, context=context))


@dataclass
class EdgeListAdapter:
    name: str = "arc_edges"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    background: int | None = None

    def __post_init__(self) -> None:
        self.requires = ["arc_grid"]
        self.provides = ["arc_edges"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        summaries: list[dict[str, Any]] = []
        for split, index, grid in _source_grids(task):
            grid = normalize_grid(grid)
            edges = edge_list(grid)
            summaries.append({"split": split, "index": index, "edge_count": len(edges)})
            _append_state(packet, value=edges, grid=grid, split=split, index=index, value_kind="edge_list")
        _arc(packet)["edges"] = summaries
        packet.log(self.name, {"examples": len(summaries), "edge_counts": [item["edge_count"] for item in summaries]}, role="adapter")
        return packet


@dataclass
class ObjectDetectorAdapter:
    name: str = "arc_objects"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    background: int | None = None
    connectivity: int = 4

    def __post_init__(self) -> None:
        self.requires = ["arc_edges"]
        self.provides = ["arc_objects"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        summaries: list[dict[str, Any]] = []
        for split, index, grid in _source_grids(task):
            grid = normalize_grid(grid)
            components = connected_components(grid, background=self.background, connectivity=self.connectivity)
            summaries.append(
                {
                    "split": split,
                    "index": index,
                    "component_count": len(components),
                    "connectivity": self.connectivity,
                }
            )
            _append_state(
                packet,
                value=tuple(asdict(component) for component in components),
                grid=grid,
                split=split,
                index=index,
                value_kind="object_detection",
                extra={"connectivity": self.connectivity},
            )
        _arc(packet)["objects"] = summaries
        packet.log(self.name, {"examples": len(summaries), "connectivity": self.connectivity}, role="adapter")
        return packet


@dataclass
class ShapeDefinerAdapter:
    name: str = "arc_shapes"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    connectivity: int = 4

    def __post_init__(self) -> None:
        self.requires = ["arc_objects"]
        self.provides = ["arc_shapes"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        summaries: list[dict[str, Any]] = []
        for split, index, grid in _source_grids(task):
            grid = normalize_grid(grid)
            shapes = shape_definitions(grid, connectivity=self.connectivity)
            summaries.append({"split": split, "index": index, "shape_count": len(shapes), "shapes": [asdict(shape) for shape in shapes]})
            _append_state(
                packet,
                value=tuple(asdict(shape) for shape in shapes),
                grid=grid,
                split=split,
                index=index,
                value_kind="shape_signature",
                extra={"connectivity": self.connectivity},
            )
        _arc(packet)["shapes"] = summaries
        packet.log(self.name, {"examples": len(summaries)}, role="adapter")
        return packet


@dataclass
class SpatialRelationAdapter:
    name: str = "arc_relations"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    connectivity: int = 4

    def __post_init__(self) -> None:
        self.requires = ["arc_shapes"]
        self.provides = ["arc_relations"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        summaries: list[dict[str, Any]] = []
        for split, index, grid in _source_grids(task):
            grid = normalize_grid(grid)
            shapes = shape_definitions(grid, connectivity=self.connectivity)
            relations = spatial_relations(shapes)
            summaries.append({"split": split, "index": index, "relation_count": len(relations), "relations": relations})
            _append_state(
                packet,
                value=relations,
                grid=grid,
                split=split,
                index=index,
                value_kind="spatial_relations",
                extra={"connectivity": self.connectivity},
            )
        _arc(packet)["relations"] = summaries
        packet.relations.extend(item["relations"] for item in summaries)
        packet.log(self.name, {"examples": len(summaries)}, role="adapter")
        return packet


@dataclass
class ColorMapperAdapter:
    name: str = "arc_colours"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_relations"]
        self.provides = ["arc_colours"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        summaries: list[dict[str, Any]] = []
        for split, index, grid in _source_grids(task):
            grid = normalize_grid(grid)
            summary = color_summary(grid)
            summaries.append({"split": split, "index": index, **summary})
            _append_state(packet, value=summary, grid=grid, split=split, index=index, value_kind="colour_summary")
        _arc(packet)["colours"] = summaries
        packet.log(self.name, {"examples": len(summaries)}, role="adapter")
        return packet


@dataclass
class PatternMinerAdapter:
    name: str = "arc_patterns"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_colours"]
        self.provides = ["arc_patterns"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        summaries: list[dict[str, Any]] = []
        for split, index, grid in _source_grids(task):
            grid = normalize_grid(grid)
            patterns = pattern_miner(grid)
            summaries.append({"split": split, "index": index, "pattern_count": len(patterns), "patterns": patterns})
            _append_state(packet, value=patterns, grid=grid, split=split, index=index, value_kind="pattern_mining")
        _arc(packet)["patterns"] = summaries
        packet.log(self.name, {"examples": len(summaries)}, role="adapter")
        return packet


@dataclass
class GridDeltaAdapter:
    name: str = "arc_grid_delta"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_patterns"]
        self.provides = ["arc_grid_delta"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        deltas: list[dict[str, Any]] = []
        for index, example in enumerate(task.train):
            if example.output_grid is None:
                continue
            delta = grid_delta(example.input_grid, example.output_grid)
            deltas.append(delta)
            packet.views.append({"name": self.name, "example_index": index, **delta})
        _arc(packet)["grid_deltas"] = deltas
        packet.deltas = deltas
        packet.log(self.name, {"deltas": len(deltas)}, role="adapter")
        return packet


@dataclass
class ObjectDeltaAdapter:
    name: str = "arc_object_delta"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_grid_delta"]
        self.provides = ["arc_object_delta"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        deltas: list[dict[str, Any]] = []
        for index, example in enumerate(task.train):
            if example.output_grid is None:
                continue
            delta = object_delta(example.input_grid, example.output_grid)
            deltas.append(delta)
            packet.views.append({"name": self.name, "example_index": index, **delta})
        _arc(packet)["object_deltas"] = deltas
        packet.log(self.name, {"deltas": len(deltas)}, role="adapter")
        return packet


@dataclass
class ColorDeltaAdapter:
    name: str = "arc_colour_delta"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_object_delta"]
        self.provides = ["arc_colour_delta"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        deltas: list[dict[str, Any]] = []
        for index, example in enumerate(task.train):
            if example.output_grid is None:
                continue
            delta = colour_delta(example.input_grid, example.output_grid)
            deltas.append(delta)
            packet.views.append({"name": self.name, "example_index": index, **delta})
        _arc(packet)["colour_deltas"] = deltas
        packet.log(self.name, {"deltas": len(deltas)}, role="adapter")
        return packet


@dataclass
class StructuralDeltaAdapter:
    name: str = "arc_structural_delta"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_colour_delta"]
        self.provides = ["arc_structural_delta"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        deltas: list[dict[str, Any]] = []
        for index, example in enumerate(task.train):
            if example.output_grid is None:
                continue
            delta = structural_delta(example.input_grid, example.output_grid)
            deltas.append(delta)
            packet.views.append({"name": self.name, "example_index": index, **delta})
        _arc(packet)["structural_deltas"] = deltas
        packet.deltas = deltas
        packet.log(self.name, {"deltas": len(deltas)}, role="adapter")
        return packet


__all__ = [
    "ColorDeltaAdapter",
    "ColorMapperAdapter",
    "EdgeListAdapter",
    "GridDeltaAdapter",
    "ObjectDetectorAdapter",
    "ObjectDeltaAdapter",
    "PatternMinerAdapter",
    "ShapeDefinerAdapter",
    "SpatialRelationAdapter",
    "StructuralDeltaAdapter",
]
