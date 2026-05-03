"""Self-contained grid helpers for adapter-layer ARC transforms."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Sequence


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


def connected_components(grid: Grid, background: int | None = None) -> list[ArcObject]:
    grid = normalize_grid(grid)
    if background is None:
        background = most_common_colour(grid)
    height = len(grid)
    width = len(grid[0]) if grid else 0
    seen: set[tuple[int, int]] = set()
    components: list[ArcObject] = []

    for row in range(height):
        for col in range(width):
            if (row, col) in seen or grid[row][col] == background:
                continue
            colour = grid[row][col]
            queue = deque([(row, col)])
            seen.add((row, col))
            cells: list[tuple[int, int]] = []

            while queue:
                current_row, current_col = queue.popleft()
                cells.append((current_row, current_col))
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if (next_row, next_col) in seen or grid[next_row][next_col] != colour:
                        continue
                    seen.add((next_row, next_col))
                    queue.append((next_row, next_col))

            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            centroid = (sum(rows) / len(cells), sum(cols) / len(cells))
            components.append(
                ArcObject(
                    colour=colour,
                    cells=tuple(sorted(cells)),
                    bbox=bbox,
                    size=len(cells),
                    centroid=centroid,
                )
            )

    return components
