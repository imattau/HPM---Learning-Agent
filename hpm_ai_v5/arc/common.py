"""Shared ARC utilities for grids, objects, and simple transformations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import ndimage as ndi


Grid = tuple[tuple[int, ...], ...]


def normalize_grid(grid: Sequence[Sequence[int]] | Grid) -> Grid:
    return tuple(tuple(int(cell) for cell in row) for row in grid)


def most_common_colour(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    if not counts:
        return 0
    return counts.most_common(1)[0][0]


def grid_context(grid: Grid) -> dict[str, Any]:
    return {
        "height": len(grid),
        "width": len(grid[0]) if grid else 0,
        "colours": sorted({cell for row in grid for cell in row}),
        "background": most_common_colour(grid),
    }


@dataclass(frozen=True, slots=True)
class ArcObject:
    colour: int
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    size: int
    centroid: tuple[float, float]


@dataclass(frozen=True, slots=True)
class ArcExample:
    input_grid: Grid
    output_grid: Grid | None = None

    @classmethod
    def from_raw(cls, raw: Any) -> "ArcExample":
        if isinstance(raw, ArcExample):
            return raw
        if isinstance(raw, dict):
            return cls(
                input_grid=normalize_grid(raw["input"]),
                output_grid=None if raw.get("output") is None else normalize_grid(raw["output"]),
            )
        input_grid, output_grid = raw
        return cls(input_grid=normalize_grid(input_grid), output_grid=normalize_grid(output_grid))


@dataclass(frozen=True, slots=True)
class ArcTask:
    train: tuple[ArcExample, ...]
    test_inputs: tuple[Grid, ...]
    task_id: str | None = None

    @classmethod
    def from_raw(cls, raw: Any) -> "ArcTask":
        if isinstance(raw, ArcTask):
            return raw
        if not isinstance(raw, dict):
            raise TypeError(f"ARC task must be a mapping or ArcTask, got {type(raw).__name__}")
        train = tuple(ArcExample.from_raw(item) for item in raw.get("train", []))
        tests: list[Grid] = []
        for item in raw.get("test", []):
            if isinstance(item, dict):
                tests.append(normalize_grid(item["input"]))
            else:
                tests.append(normalize_grid(item))
        return cls(train=train, test_inputs=tuple(tests), task_id=raw.get("task_id"))


def connected_components(grid: Grid, background: int | None = None, connectivity: int = 4) -> list[ArcObject]:
    grid = normalize_grid(grid)
    if background is None:
        background = most_common_colour(grid)
    height = len(grid)
    width = len(grid[0]) if grid else 0
    components: list[ArcObject] = []

    if height == 0 or width == 0:
        return components

    structure = np.ones((3, 3), dtype=int) if connectivity == 8 else np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        dtype=int,
    )
    arr = np.array(grid, dtype=int)
    colours = [colour for colour in sorted(set(arr.flatten().tolist())) if colour != background]

    for colour in colours:
        labels, count = ndi.label(arr == colour, structure=structure)
        for label_index in range(1, count + 1):
            coords = np.argwhere(labels == label_index)
            if coords.size == 0:
                continue
            cells = tuple((int(row), int(col)) for row, col in coords.tolist())
            rows = coords[:, 0]
            cols = coords[:, 1]
            bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
            centroid = (float(rows.mean()), float(cols.mean()))
            components.append(
                ArcObject(
                    colour=int(colour),
                    cells=tuple(sorted(cells)),
                    bbox=bbox,
                    size=len(cells),
                    centroid=centroid,
                )
            )

    return components


def largest_object(grid: Grid, background: int | None = None) -> ArcObject | None:
    components = connected_components(grid, background=background)
    if not components:
        return None
    return max(components, key=lambda item: item.size)


def _transform_point(kind: str, row: int, col: int, height: int, width: int) -> tuple[int, int]:
    if kind == "identity":
        return row, col
    if kind == "translate":
        return row, col
    if kind == "rotate_90":
        return col, height - 1 - row
    if kind == "rotate_180":
        return height - 1 - row, width - 1 - col
    if kind == "rotate_270":
        return width - 1 - col, row
    if kind == "mirror_horizontal":
        return row, width - 1 - col
    if kind == "mirror_vertical":
        return height - 1 - row, col
    raise ValueError(f"Unknown ARC transform kind: {kind}")


def _line_points(axis: str, index: int, length: int) -> list[tuple[int, int]]:
    if axis == "row":
        return [(index, col) for col in range(length)]
    if axis == "col":
        return [(row, index) for row in range(length)]
    raise ValueError(f"Unknown ARC line axis: {axis}")


def _crop(grid: Grid, bbox: tuple[int, int, int, int]) -> Grid:
    top, left, bottom, right = bbox
    height = len(grid)
    width = len(grid[0]) if grid else 0
    if top < 0 or left < 0 or bottom >= height or right >= width or top > bottom or left > right:
        return tuple()
    return tuple(tuple(grid[row][col] for col in range(left, right + 1)) for row in range(top, bottom + 1))


@dataclass(frozen=True, slots=True)
class ArcTransformation:
    kind: str = "identity"
    axis: str | None = None
    line_index: int = 0
    crop_bbox: tuple[int, int, int, int] | None = None
    preserve_canvas: bool = False
    dx: int = 0
    dy: int = 0
    colour_map: dict[int, int] = field(default_factory=dict)
    background: int = 0
    label: str = "translate_recolour"

    def complexity(self) -> int:
        kind_cost = 0 if self.kind in {"identity", "translate"} else 1
        if self.kind == "extend_line":
            kind_cost = 1
        if self.kind == "crop_object":
            kind_cost = 1
        return kind_cost + abs(self.dx) + abs(self.dy) + len(self.colour_map)

    def apply(self, grid: Sequence[Sequence[int]] | Grid) -> Grid:
        grid = normalize_grid(grid)
        height = len(grid)
        width = len(grid[0]) if grid else 0
        background = self.background
        output = [[background for _ in range(width)] for _ in range(height)]
        if self.kind == "crop_object" and self.crop_bbox is not None:
            cropped = _crop(grid, self.crop_bbox)
            crop_height = len(cropped)
            crop_width = len(cropped[0]) if cropped else 0
            if self.preserve_canvas:
                for row_index, row in enumerate(cropped):
                    for col_index, colour in enumerate(row):
                        target_row = self.crop_bbox[0] + row_index
                        target_col = self.crop_bbox[1] + col_index
                        if 0 <= target_row < height and 0 <= target_col < width:
                            output[target_row][target_col] = self.colour_map.get(colour, colour)
            else:
                output = [[background for _ in range(crop_width)] for _ in range(crop_height)]
                for row_index, row in enumerate(cropped):
                    for col_index, colour in enumerate(row):
                        output[row_index][col_index] = self.colour_map.get(colour, colour)
            return normalize_grid(output)
        if self.kind == "extend_line" and self.axis is not None:
            for row_index, row in enumerate(grid):
                for col_index, colour in enumerate(row):
                    if colour != background:
                        output[row_index][col_index] = self.colour_map.get(colour, colour)
            line_colour = next(iter(self.colour_map.keys()), None)
            if line_colour is None:
                return normalize_grid(output)
            target_colour = self.colour_map.get(line_colour, line_colour)
            points = _line_points(self.axis, self.line_index, width if self.axis == "row" else height)
            for row_index, col_index in points:
                output[row_index][col_index] = target_colour
            return normalize_grid(output)
        for row_index, row in enumerate(grid):
            for col_index, colour in enumerate(row):
                if colour == background:
                    continue
                next_row, next_col = _transform_point(self.kind, row_index, col_index, height, width)
                next_row += self.dy
                next_col += self.dx
                if 0 <= next_row < height and 0 <= next_col < width:
                    output[next_row][next_col] = self.colour_map.get(colour, colour)
        return normalize_grid(output)

    def matches(self, input_grid: Sequence[Sequence[int]] | Grid, output_grid: Sequence[Sequence[int]] | Grid) -> bool:
        return self.apply(input_grid) == normalize_grid(output_grid)

    def describe(self) -> dict[str, Any]:
        return {
            "dx": self.dx,
            "dy": self.dy,
            "colour_map": dict(self.colour_map),
            "background": self.background,
            "preserve_canvas": self.preserve_canvas,
            "axis": self.axis,
            "line_index": self.line_index,
            "crop_bbox": self.crop_bbox,
            "kind": self.kind,
            "label": self.label,
            "complexity": self.complexity(),
        }


def infer_transformation(input_grid: Sequence[Sequence[int]] | Grid, output_grid: Sequence[Sequence[int]] | Grid) -> ArcTransformation | None:
    input_grid = normalize_grid(input_grid)
    output_grid = normalize_grid(output_grid)
    if not input_grid or not output_grid:
        return None
    input_background = most_common_colour(input_grid)
    output_background = most_common_colour(output_grid)
    height = len(input_grid)
    width = len(input_grid[0]) if input_grid else 0
    foreground = [(row, col, input_grid[row][col]) for row in range(height) for col in range(width) if input_grid[row][col] != input_background]
    if not foreground:
        return None

    kinds = ("identity", "translate", "rotate_90", "rotate_180", "rotate_270", "mirror_horizontal", "mirror_vertical")
    output_foreground = {(row, col): output_grid[row][col] for row in range(len(output_grid)) for col in range(len(output_grid[0])) if output_grid[row][col] != output_background}

    for kind in kinds:
        transformed = [_transform_point(kind, row, col, height, width) for row, col, _ in foreground]
        in_rows = [row for row, _ in transformed]
        in_cols = [col for _, col in transformed]
        out_rows = [row for row, col in output_foreground]
        out_cols = [col for row, col in output_foreground]
        if not in_rows or not out_rows:
            continue

        dx = int(round((sum(out_cols) / len(out_cols)) - (sum(in_cols) / len(in_cols))))
        dy = int(round((sum(out_rows) / len(out_rows)) - (sum(in_rows) / len(in_rows))))

        colour_map: dict[int, int] = {}
        valid = True
        transformed_set: set[tuple[int, int]] = set()
        for (row, col, colour), (next_row, next_col) in zip(foreground, transformed):
            next_row += dy
            next_col += dx
            if (next_row, next_col) not in output_foreground:
                valid = False
                break
            transformed_set.add((next_row, next_col))
            output_colour = output_foreground[(next_row, next_col)]
            existing = colour_map.get(colour)
            if existing is not None and existing != output_colour:
                valid = False
                break
            colour_map[colour] = output_colour

        if valid and len(transformed_set) == len(output_foreground):
            resolved_kind = "translate" if kind == "identity" and (dx != 0 or dy != 0) else kind
            resolved_label = "translate_recolour" if resolved_kind == "translate" else f"{resolved_kind}_recolour"
            return ArcTransformation(kind=resolved_kind, dx=dx, dy=dy, colour_map=colour_map, background=output_background, label=resolved_label)

    input_colours = {colour for _, _, colour in foreground}
    output_colours = {colour for colour in output_foreground.values()}
    if len(input_colours) == 1 and len(output_colours) == 1:
        input_colour = next(iter(input_colours))
        output_colour = next(iter(output_colours))
        input_rows = {row for row, _, _ in foreground}
        input_cols = {col for _, col, _ in foreground}
        output_rows = {row for row, _ in output_foreground}
        output_cols = {col for _, col in output_foreground}
        if len(input_rows) == 1 and len(output_rows) == 1:
            row_index = next(iter(output_rows))
            input_span = [col for _, col, _ in foreground]
            output_span = [col for row, col in output_foreground if row == row_index]
            if input_span and output_span and min(output_span) <= min(input_span) and max(output_span) >= max(input_span):
                return ArcTransformation(
                    kind="extend_line",
                    axis="row",
                    line_index=row_index,
                    colour_map={input_colour: output_colour},
                    background=output_background,
                    label="extend_line",
                )
        if len(input_cols) == 1 and len(output_cols) == 1:
            col_index = next(iter(output_cols))
            input_span = [row for row, _, _ in foreground]
            output_span = [row for row, col in output_foreground if col == col_index]
            if input_span and output_span and min(output_span) <= min(input_span) and max(output_span) >= max(input_span):
                return ArcTransformation(
                    kind="extend_line",
                    axis="col",
                    line_index=col_index,
                    colour_map={input_colour: output_colour},
                    background=output_background,
                    label="extend_line",
                )

    input_object = largest_object(input_grid, background=input_background)
    if input_object is not None:
        cropped = _crop(input_grid, input_object.bbox)
        preserve_canvas = len(output_grid) == len(input_grid) and len(output_grid[0]) == len(input_grid[0])
        if preserve_canvas:
            if len(output_grid) == len(input_grid) and len(output_grid[0]) == len(input_grid[0]):
                colour_map: dict[int, int] = {}
                valid = True
                for row_index, row in enumerate(cropped):
                    for col_index, colour in enumerate(row):
                        output_row = input_object.bbox[0] + row_index
                        output_col = input_object.bbox[1] + col_index
                        output_colour = output_grid[output_row][output_col]
                        existing = colour_map.get(colour)
                        if existing is not None and existing != output_colour:
                            valid = False
                            break
                        colour_map[colour] = output_colour
                    if not valid:
                        break
                if valid:
                    return ArcTransformation(
                        kind="crop_object",
                        crop_bbox=input_object.bbox,
                        preserve_canvas=True,
                        colour_map=colour_map,
                        background=output_background,
                        label="crop_object",
                    )
        if len(cropped) == len(output_grid) and len(cropped[0]) == len(output_grid[0]):
            colour_map: dict[int, int] = {}
            valid = True
            for row_index, row in enumerate(cropped):
                for col_index, colour in enumerate(row):
                    output_colour = output_grid[row_index][col_index]
                    if colour == input_background and output_colour == output_background:
                        continue
                    existing = colour_map.get(colour)
                    if existing is not None and existing != output_colour:
                        valid = False
                        break
                    colour_map[colour] = output_colour
                if not valid:
                    break
            if valid:
                return ArcTransformation(
                    kind="crop_object",
                    crop_bbox=input_object.bbox,
                    preserve_canvas=False,
                    colour_map=colour_map,
                    background=output_background,
                    label="crop_object",
                )

    return None


def merge_transformations(transformations: Iterable[ArcTransformation | None]) -> ArcTransformation | None:
    candidates = [item for item in transformations if item is not None]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    kind = max(set(item.kind for item in candidates), key=lambda value: sum(1 for item in candidates if item.kind == value))
    dx = max(set(item.dx for item in candidates), key=lambda value: sum(1 for item in candidates if item.dx == value))
    dy = max(set(item.dy for item in candidates), key=lambda value: sum(1 for item in candidates if item.dy == value))
    background = max(set(item.background for item in candidates), key=lambda value: sum(1 for item in candidates if item.background == value))
    axis = max(set(item.axis for item in candidates), key=lambda value: sum(1 for item in candidates if item.axis == value))
    line_index = max(set(item.line_index for item in candidates), key=lambda value: sum(1 for item in candidates if item.line_index == value))
    crop_bbox = max(set(item.crop_bbox for item in candidates), key=lambda value: sum(1 for item in candidates if item.crop_bbox == value))

    shared: dict[int, int] = {}
    for colour in {colour for item in candidates for colour in item.colour_map}:
        values = [item.colour_map[colour] for item in candidates if colour in item.colour_map]
        if values and all(value == values[0] for value in values):
            shared[colour] = values[0]

    resolved_kind = "translate" if kind == "identity" and (dx != 0 or dy != 0) else kind
    resolved_label = "translate_recolour" if resolved_kind == "translate" else f"{resolved_kind}_recolour"
    if resolved_kind == "extend_line":
        return ArcTransformation(kind=resolved_kind, axis=axis, line_index=line_index, colour_map=shared, background=background, label="extend_line")
    if resolved_kind == "crop_object":
        return ArcTransformation(kind=resolved_kind, crop_bbox=crop_bbox, colour_map=shared, background=background, label="crop_object")
    return ArcTransformation(kind=resolved_kind, dx=dx, dy=dy, colour_map=shared, background=background, label=resolved_label)


def score_transformation(transformation: ArcTransformation, examples: Sequence[ArcExample]) -> dict[str, float]:
    exact = 0
    for example in examples:
        if example.output_grid is None:
            continue
        try:
            if transformation.matches(example.input_grid, example.output_grid):
                exact += 1
        except (IndexError, ValueError):
            continue
    train_accuracy = exact / max(1, len(examples))
    simplicity = 1.0 / (1.0 + transformation.complexity())
    consistency = 1.0 if exact == len(examples) else train_accuracy
    score = (4.0 * train_accuracy) + (2.0 * consistency) + simplicity - (0.1 * transformation.complexity())
    return {
        "train_accuracy": train_accuracy,
        "simplicity": simplicity,
        "consistency": consistency,
        "score": score,
    }
