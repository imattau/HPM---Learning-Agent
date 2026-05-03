"""ARC preprocessing adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ...adapter import AdapterPacket
from ...core import State
from ..common import ArcExample, ArcTask, connected_components, grid_context, infer_transformation, normalize_grid


def _arc(packet: AdapterPacket) -> dict[str, Any]:
    return packet.context.setdefault("arc", {})


@dataclass
class TaskAdapter:
    name: str = "arc_task"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = []
        self.provides = ["arc_task"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task = ArcTask.from_raw(packet.raw)
        arc = _arc(packet)
        arc["task"] = task
        packet.log(self.name, {"task_id": task.task_id, "train": len(task.train), "test": len(task.test_inputs)}, role="adapter")
        return packet


@dataclass
class GridAdapter:
    name: str = "arc_grid"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_task"]
        self.provides = ["arc_grid"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task = _arc(packet)["task"]
        packet.states.clear()
        for index, example in enumerate(task.train):
            packet.states.append(State(value=example.input_grid, context={"split": "train", "index": index, **grid_context(example.input_grid)}))
        for index, grid in enumerate(task.test_inputs):
            packet.states.append(State(value=grid, context={"split": "test", "index": index, **grid_context(grid)}))
        packet.log(self.name, {"states": len(packet.states)}, role="adapter")
        return packet


@dataclass
class ObjectExtractionAdapter:
    name: str = "arc_objects"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_grid"]
        self.provides = ["arc_objects"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        views: list[dict[str, Any]] = []
        for index, example in enumerate(task.train):
            views.append(
                {
                    "name": "object_polygraph",
                    "example_index": index,
                    "objects": [asdict(obj) for obj in connected_components(example.input_grid)],
                }
            )
        packet.views.extend(views)
        packet.log(self.name, {"views": len(views)}, role="adapter")
        return packet


@dataclass
class ColourMapAdapter:
    name: str = "arc_colours"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_objects"]
        self.provides = ["arc_colours"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        views: list[dict[str, Any]] = []
        for index, example in enumerate(task.train):
            if example.output_grid is None:
                continue
            candidates = infer_transformation(example.input_grid, example.output_grid)
            if candidates is None:
                continue
            views.append({"name": "colour_polygraph", "example_index": index, "colour_map": dict(candidates.colour_map)})
        packet.views.extend(views)
        packet.log(self.name, {"views": len(views)}, role="adapter")
        return packet


@dataclass
class GeometryAdapter:
    name: str = "arc_geometry"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_colours"]
        self.provides = ["arc_geometry"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        views: list[dict[str, Any]] = []
        for index, example in enumerate(task.train):
            if example.output_grid is None:
                continue
            candidate = infer_transformation(example.input_grid, example.output_grid)
            if candidate is None:
                continue
            views.append({"name": "geometry_polygraph", "example_index": index, "dx": candidate.dx, "dy": candidate.dy})
        packet.views.extend(views)
        packet.log(self.name, {"views": len(views)}, role="adapter")
        return packet


@dataclass
class DeltaAdapter:
    name: str = "arc_delta"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_geometry"]
        self.provides = ["arc_delta"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        deltas: list[Any] = []
        for example in task.train:
            if example.output_grid is None:
                continue
            deltas.append(infer_transformation(example.input_grid, example.output_grid))
        packet.deltas = deltas
        packet.log(self.name, {"deltas": len(deltas)}, role="adapter")
        return packet


@dataclass
class ExamplePairAdapter:
    name: str = "arc_examples"
    requires: list[str] = None  # type: ignore[assignment]
    provides: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.requires = ["arc_line_extension", "arc_structural_delta"]
        self.provides = ["arc_examples"]

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        task: ArcTask = _arc(packet)["task"]
        arc = _arc(packet)
        deltas = packet.deltas or arc.get("structural_deltas", [])
        arc["examples"] = [
            {
                "input": example.input_grid,
                "output": example.output_grid,
                "delta": None if index >= len(deltas) else deltas[index],
            }
            for index, example in enumerate(task.train)
        ]
        packet.log(self.name, {"examples": len(task.train)}, role="adapter")
        return packet
