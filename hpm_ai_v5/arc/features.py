"""Higher-level ARC structural features."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
from skimage import measure

from .common import ArcObject, Grid, connected_components, grid_context, most_common_colour, normalize_grid


@dataclass(frozen=True, slots=True)
class ArcShape:
    colour: int
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    size: int
    centroid: tuple[float, float]
    perimeter: int
    holes: int
    symmetry: float
    aspect_ratio: float
    solidity: float
    kind: str
    fill_ratio: float


def edge_list(grid: Sequence[Sequence[int]] | Grid) -> tuple[tuple[int, int, int, int, int, int, int], ...]:
    grid = normalize_grid(grid)
    height = len(grid)
    width = len(grid[0]) if grid else 0
    graph = nx.grid_2d_graph(height, width)
    for row in range(height):
        for col in range(width):
            graph.nodes[(row, col)]["value"] = grid[row][col]
    edges: list[tuple[int, int, int, int, int, int, int]] = []
    for (row1, col1), (row2, col2) in sorted(graph.edges()):
        value1 = int(graph.nodes[(row1, col1)]["value"])
        value2 = int(graph.nodes[(row2, col2)]["value"])
        edges.append((row1, col1, row2, col2, value1, value2, abs(value2 - value1)))
    return tuple(edges)


def color_summary(grid: Sequence[Sequence[int]] | Grid) -> dict[str, Any]:
    grid = normalize_grid(grid)
    counts = Counter(cell for row in grid for cell in row)
    total = sum(counts.values()) or 1
    background = most_common_colour(grid)
    return {
        "background": background,
        "palette": tuple(sorted(counts)),
        "counts": {colour: count for colour, count in counts.items()},
        "frequencies": {colour: count / total for colour, count in counts.items()},
    }


def _perimeter(cells: Iterable[tuple[int, int]]) -> int:
    cells_set = set(cells)
    perimeter = 0
    for row, col in cells_set:
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if (next_row, next_col) not in cells_set:
                perimeter += 1
    return perimeter


def _shape_metrics(cells: tuple[tuple[int, int], ...], bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    top, left, bottom, right = bbox
    height = bottom - top + 1
    width = right - left + 1
    mask = np.zeros((height, width), dtype=bool)
    for row, col in cells:
        mask[row - top, col - left] = True

    labelled = measure.label(mask.astype(int), connectivity=1)
    region = measure.regionprops(labelled)[0]
    euler_number = int(region.euler_number)
    holes = max(0, 1 - euler_number)
    perimeter = int(round(float(region.perimeter or 0.0)))
    fill_ratio = float(region.extent)
    solidity = float(region.solidity)
    symmetry = float(
        (
            np.mean(mask == np.flipud(mask))
            + np.mean(mask == np.fliplr(mask))
        )
        / 2.0
    )
    aspect_ratio = float(max(height, width) / max(1, min(height, width)))
    return {
        "perimeter": perimeter,
        "holes": holes,
        "symmetry": symmetry,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "fill_ratio": fill_ratio,
    }


def _region_kind(metrics: dict[str, Any]) -> str:
    if metrics["holes"] > 0:
        return "holed_region"
    if metrics["solidity"] >= 0.95:
        return "filled_region"
    if metrics["symmetry"] >= 0.9:
        return "symmetric_region"
    if metrics["aspect_ratio"] >= 3.0:
        return "elongated_region"
    if metrics["fill_ratio"] <= 0.35:
        return "sparse_region"
    return "structured_region"


def shape_definitions(
    grid: Sequence[Sequence[int]] | Grid,
    *,
    background: int | None = None,
    connectivity: int = 4,
) -> tuple[ArcShape, ...]:
    grid = normalize_grid(grid)
    components = connected_components(grid, background=background, connectivity=connectivity)
    shapes: list[ArcShape] = []
    for component in components:
        bbox = component.bbox
        metrics = _shape_metrics(component.cells, bbox)
        shapes.append(
            ArcShape(
                colour=component.colour,
                cells=component.cells,
                bbox=bbox,
                size=component.size,
                centroid=component.centroid,
                perimeter=metrics["perimeter"],
                holes=metrics["holes"],
                symmetry=metrics["symmetry"],
                aspect_ratio=metrics["aspect_ratio"],
                solidity=metrics["solidity"],
                kind=_region_kind(metrics),
                fill_ratio=metrics["fill_ratio"],
            )
        )
    return tuple(shapes)


def spatial_relations(objects: Sequence[ArcShape | ArcObject]) -> tuple[dict[str, Any], ...]:
    relations: list[dict[str, Any]] = []
    for left_index, left_object in enumerate(objects):
        left_bbox = left_object.bbox
        left_top, left_left, left_bottom, left_right = left_bbox
        left_centroid = getattr(left_object, "centroid")
        for right_index, right_object in enumerate(objects):
            if left_index == right_index:
                continue
            right_top, right_left, right_bottom, right_right = right_object.bbox
            right_centroid = getattr(right_object, "centroid")
            relation = "overlaps"
            if left_top <= right_top and left_left <= right_left and left_bottom >= right_bottom and left_right >= right_right:
                relation = "contains"
            elif right_top <= left_top and right_left <= left_left and right_bottom >= left_bottom and right_right >= left_right:
                relation = "inside"
            elif left_bottom < right_top:
                relation = "above"
            elif left_top > right_bottom:
                relation = "below"
            elif left_right < right_left:
                relation = "left_of"
            elif left_left > right_right:
                relation = "right_of"
            relations.append(
                {
                    "left": left_index,
                    "right": right_index,
                    "relation": relation,
                    "left_centroid": left_centroid,
                    "right_centroid": right_centroid,
                }
            )
    return tuple(relations)


def relation_graph(objects: Sequence[ArcShape | ArcObject]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for index, obj in enumerate(objects):
        graph.add_node(
            index,
            colour=obj.colour,
            bbox=obj.bbox,
            size=obj.size,
            centroid=obj.centroid,
        )
    for relation in spatial_relations(objects):
        graph.add_edge(
            relation["left"],
            relation["right"],
            relation=relation["relation"],
            left_centroid=relation["left_centroid"],
            right_centroid=relation["right_centroid"],
        )
    return graph


def grid_delta(
    input_grid: Sequence[Sequence[int]] | Grid,
    output_grid: Sequence[Sequence[int]] | Grid,
) -> dict[str, Any]:
    input_grid = normalize_grid(input_grid)
    output_grid = normalize_grid(output_grid)
    delta_grid: tuple[tuple[int, ...], ...] | None = None
    if input_grid and output_grid and len(input_grid) == len(output_grid) and len(input_grid[0]) == len(output_grid[0]):
        delta_grid = tuple(
            tuple(output_grid[row][col] - input_grid[row][col] for col in range(len(input_grid[0])))
            for row in range(len(input_grid))
        )
    changed_cells = tuple(
        (row, col, input_grid[row][col], output_grid[row][col])
        for row in range(min(len(input_grid), len(output_grid)))
        for col in range(min(len(input_grid[0]) if input_grid else 0, len(output_grid[0]) if output_grid else 0))
        if input_grid[row][col] != output_grid[row][col]
    )
    return {
        "input_shape": (len(input_grid), len(input_grid[0]) if input_grid else 0),
        "output_shape": (len(output_grid), len(output_grid[0]) if output_grid else 0),
        "delta_grid": delta_grid,
        "changed_cells": changed_cells,
    }


def colour_delta(
    input_grid: Sequence[Sequence[int]] | Grid,
    output_grid: Sequence[Sequence[int]] | Grid,
) -> dict[str, Any]:
    input_grid = normalize_grid(input_grid)
    output_grid = normalize_grid(output_grid)
    input_counts = Counter(cell for row in input_grid for cell in row)
    output_counts = Counter(cell for row in output_grid for cell in row)
    substitutions: dict[int, int] = {}
    if input_grid and output_grid and len(input_grid) == len(output_grid) and len(input_grid[0]) == len(output_grid[0]):
        for row in range(len(input_grid)):
            for col in range(len(input_grid[0])):
                source = input_grid[row][col]
                target = output_grid[row][col]
                if source == target:
                    continue
                existing = substitutions.get(source)
                if existing is not None and existing != target:
                    continue
                substitutions[source] = target
    palette = sorted(set(input_counts) | set(output_counts))
    return {
        "input_palette": tuple(sorted(input_counts)),
        "output_palette": tuple(sorted(output_counts)),
        "count_delta": {colour: output_counts.get(colour, 0) - input_counts.get(colour, 0) for colour in palette},
        "substitutions": substitutions,
    }


def object_delta(
    input_grid: Sequence[Sequence[int]] | Grid,
    output_grid: Sequence[Sequence[int]] | Grid,
    *,
    background: int | None = None,
) -> dict[str, Any]:
    input_objects = shape_definitions(input_grid, background=background)
    output_objects = shape_definitions(output_grid, background=background)
    exact_input = Counter((obj.colour, obj.bbox, obj.kind, obj.size) for obj in input_objects)
    exact_output = Counter((obj.colour, obj.bbox, obj.kind, obj.size) for obj in output_objects)
    loose_input = Counter((obj.colour, obj.kind, obj.size) for obj in input_objects)
    loose_output = Counter((obj.colour, obj.kind, obj.size) for obj in output_objects)

    added = list((exact_output - exact_input).elements())
    removed = list((exact_input - exact_output).elements())
    moved: list[dict[str, Any]] = []
    for obj in input_objects:
        if (obj.colour, obj.kind, obj.size) not in loose_output:
            continue
        matching = [other for other in output_objects if (other.colour, other.kind, other.size) == (obj.colour, obj.kind, obj.size)]
        for other in matching:
            if other.bbox != obj.bbox:
                moved.append({"colour": obj.colour, "kind": obj.kind, "size": obj.size, "from": obj.bbox, "to": other.bbox})
                break

    return {
        "input_objects": tuple(asdict(obj) for obj in input_objects),
        "output_objects": tuple(asdict(obj) for obj in output_objects),
        "added": added,
        "removed": removed,
        "moved": tuple(moved),
    }


def structural_delta(
    input_grid: Sequence[Sequence[int]] | Grid,
    output_grid: Sequence[Sequence[int]] | Grid,
    *,
    background: int | None = None,
) -> dict[str, Any]:
    input_shapes = shape_definitions(input_grid, background=background)
    output_shapes = shape_definitions(output_grid, background=background)
    input_relations = spatial_relations(input_shapes)
    output_relations = spatial_relations(output_shapes)
    input_kind_counts = Counter(shape.kind for shape in input_shapes)
    output_kind_counts = Counter(shape.kind for shape in output_shapes)
    all_kinds = sorted(set(input_kind_counts) | set(output_kind_counts))
    return {
        "input": {
            "component_count": len(input_shapes),
            "shapes": tuple(asdict(shape) for shape in input_shapes),
            "relations": input_relations,
        },
        "output": {
            "component_count": len(output_shapes),
            "shapes": tuple(asdict(shape) for shape in output_shapes),
            "relations": output_relations,
        },
        "component_count_delta": len(output_shapes) - len(input_shapes),
        "shape_kind_delta": {kind: output_kind_counts.get(kind, 0) - input_kind_counts.get(kind, 0) for kind in all_kinds},
    }


def pattern_miner(
    grid: Sequence[Sequence[int]] | Grid,
    *,
    sizes: Sequence[int] = (2, 3),
) -> tuple[dict[str, Any], ...]:
    grid = normalize_grid(grid)
    height = len(grid)
    width = len(grid[0]) if grid else 0
    mined: list[dict[str, Any]] = []
    for size in sizes:
        if size <= 0 or height < size or width < size:
            continue
        occurrences: dict[tuple[tuple[int, ...], ...], list[tuple[int, int]]] = {}
        for row in range(height - size + 1):
            for col in range(width - size + 1):
                pattern = tuple(tuple(grid[row + r][col + c] for c in range(size)) for r in range(size))
                occurrences.setdefault(pattern, []).append((row, col))
        for pattern, positions in occurrences.items():
            if len(positions) < 2:
                continue
            mined.append(
                {
                    "size": size,
                    "pattern": pattern,
                    "count": len(positions),
                    "positions": tuple(positions),
                }
            )
    mined.sort(key=lambda item: (-item["count"], -item["size"]))
    return tuple(mined)


__all__ = [
    "ArcShape",
    "colour_delta",
    "color_summary",
    "edge_list",
    "grid_delta",
    "object_delta",
    "pattern_miner",
    "relation_graph",
    "shape_definitions",
    "spatial_relations",
    "structural_delta",
]
