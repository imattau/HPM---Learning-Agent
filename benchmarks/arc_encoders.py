"""ARC-specific LevelEncoder implementations for the structured ARC benchmark.

ObjectParser: 4-connected component labelling per colour (background=0 ignored).
ObjectMatcher: colour-first matching, then greedy centroid proximity.
ArcL1Encoder: pixel delta projected to 64-dim (same as multi_agent_arc.py).
ArcL2Encoder: per-matched-pair 9-dim object anatomy vector.
ArcL3Encoder: 14-dim relational transformation summary.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from dataclasses import dataclass
import numpy as np

MAX_GRID_DIM = 20
GRID_SIZE = MAX_GRID_DIM * MAX_GRID_DIM  # 400
MAX_OBJ = 20  # normalisation for object counts
L1_DIM = 64
L2_DIM = 9
L3_DIM = 14

# Fixed random projection matrix (same seed as multi_agent_arc.py)
_L1_PROJ = np.random.default_rng(0).standard_normal((GRID_SIZE, L1_DIM)) / np.sqrt(L1_DIM)


@dataclass
class ArcObject:
    id: int
    color: int
    bbox: tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    area: int
    perimeter: int
    centroid: tuple[float, float]    # (row, col)


def parse_objects(grid: list[list[int]]) -> list[ArcObject]:
    """Extract 4-connected objects. Returns list sorted by area desc, then centroid row-major."""
    g = np.array(grid, dtype=int)
    rows, cols = g.shape
    visited = np.zeros_like(g, dtype=bool)
    objects: list[ArcObject] = []
    obj_id = 0

    for r in range(rows):
        for c in range(cols):
            color = int(g[r, c])
            if color == 0 or visited[r, c]:
                continue
            # BFS
            cells: list[tuple[int, int]] = []
            queue = [(r, c)]
            visited[r, c] = True
            while queue:
                cr, cc = queue.pop(0)
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and g[nr, nc] == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            rs = [cell[0] for cell in cells]
            cs_ = [cell[1] for cell in cells]
            min_r, max_r = min(rs), max(rs)
            min_c, max_c = min(cs_), max(cs_)
            area = len(cells)
            centroid = (sum(rs) / area, sum(cs_) / area)

            # Perimeter: boundary edges
            cell_set = set(cells)
            perim = sum(
                1
                for cr, cc in cells
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if (cr + dr, cc + dc) not in cell_set
            )

            objects.append(ArcObject(
                id=obj_id, color=color,
                bbox=(min_r, min_c, max_r, max_c),
                area=area, perimeter=perim, centroid=centroid,
            ))
            obj_id += 1

    objects.sort(key=lambda o: (-o.area, o.centroid[0], o.centroid[1]))
    return objects


@dataclass
class MatchedPair:
    input_obj: ArcObject
    output_obj: ArcObject


def match_objects(
    input_objects: list[ArcObject],
    output_objects: list[ArcObject],
) -> tuple[list[MatchedPair], list[ArcObject], list[ArcObject]]:
    """Match input objects to output objects.

    Strategy:
      1. Colour-first: unique colour in both input and output -> direct match.
      2. Remaining unmatched: greedy centroid proximity, iterated largest-area-first.

    Returns:
        (matched_pairs, appeared, disappeared)
        appeared: objects in output with no input match.
        disappeared: objects in input with no output match.
    """
    in_by_color: dict[int, list[ArcObject]] = {}
    for obj in input_objects:
        in_by_color.setdefault(obj.color, []).append(obj)
    out_by_color: dict[int, list[ArcObject]] = {}
    for obj in output_objects:
        out_by_color.setdefault(obj.color, []).append(obj)

    matched: list[MatchedPair] = []
    unmatched_in = list(input_objects)
    unmatched_out = list(output_objects)

    # Pass 1: unique-colour matching
    for color, in_objs in list(in_by_color.items()):
        out_objs = out_by_color.get(color, [])
        if len(in_objs) == 1 and len(out_objs) == 1:
            matched.append(MatchedPair(in_objs[0], out_objs[0]))
            unmatched_in.remove(in_objs[0])
            unmatched_out.remove(out_objs[0])

    # Pass 2: greedy centroid (unmatched_in already area-sorted by parse_objects)
    for in_obj in list(unmatched_in):
        if not unmatched_out:
            break
        best_dist = float("inf")
        best_out: ArcObject | None = None
        for out_obj in unmatched_out:
            dr = in_obj.centroid[0] - out_obj.centroid[0]
            dc = in_obj.centroid[1] - out_obj.centroid[1]
            dist = math.sqrt(dr * dr + dc * dc)
            if dist < best_dist:
                best_dist = dist
                best_out = out_obj
        if best_out is not None:
            matched.append(MatchedPair(in_obj, best_out))
            unmatched_in.remove(in_obj)
            unmatched_out.remove(best_out)

    appeared = unmatched_out
    disappeared = unmatched_in
    return matched, appeared, disappeared


def _encode_grid_flat(grid: list[list[int]]) -> np.ndarray:
    flat: list[float] = []
    for row in grid:
        for val in row:
            flat.append(float(val) / 9.0)
    flat.extend([0.0] * (GRID_SIZE - len(flat)))
    return np.array(flat[:GRID_SIZE], dtype=float)


class ArcL1Encoder:
    """Pixel delta encoder: (output_flat - input_flat) @ random_projection -> 64-dim."""
    feature_dim: int = L1_DIM
    max_steps_per_obs: int | None = 1

    def encode(self, observation, epistemic=None) -> list[np.ndarray]:
        input_grid, output_grid = observation
        delta = _encode_grid_flat(output_grid) - _encode_grid_flat(input_grid)
        return [delta @ _L1_PROJ]


class ArcL2Encoder:
    """Per-object-pair anatomy encoder: 9-dim vector per matched pair.

    Appeared and disappeared objects are excluded (contribute nothing to L2 score).
    All spatial values normalised by MAX_GRID_DIM=20 (fixed cross-task denominator).
    """
    feature_dim: int = L2_DIM
    max_steps_per_obs: int | None = None  # variable -- one per matched pair

    def encode(self, observation, epistemic=None) -> list[np.ndarray]:
        input_grid, output_grid = observation
        l1_weight = epistemic[0] if epistemic is not None else 0.0
        l1_loss = epistemic[1] if epistemic is not None else 0.0

        in_objs = parse_objects(input_grid)
        out_objs = parse_objects(output_grid)
        matched, _appeared, _disappeared = match_objects(in_objs, out_objs)

        vecs: list[np.ndarray] = []
        for pair in matched:
            out = pair.output_obj
            min_r, min_c, max_r, max_c = out.bbox
            vec = np.array([
                min_r / MAX_GRID_DIM,
                min_c / MAX_GRID_DIM,
                max_r / MAX_GRID_DIM,
                max_c / MAX_GRID_DIM,
                out.area / (MAX_GRID_DIM ** 2),
                out.perimeter / (4 * MAX_GRID_DIM),
                out.color / 9.0,
                l1_weight,
                l1_loss,
            ], dtype=float)
            vecs.append(vec)
        return vecs


class ArcL3Encoder:
    """Relational transformation encoder: 14-dim summary of the full transformation.

    Captures: object counts, movement/recolouring indicators, topology changes,
    mean positional delta, consistency flags, and L2 epistemic state.
    """
    feature_dim: int = L3_DIM
    max_steps_per_obs: int | None = 1

    def encode(self, observation, epistemic=None) -> list[np.ndarray]:
        input_grid, output_grid = observation
        l2_weight = epistemic[0] if epistemic is not None else 0.0
        l2_loss = epistemic[1] if epistemic is not None else 0.0

        in_objs = parse_objects(input_grid)
        out_objs = parse_objects(output_grid)
        matched, appeared, disappeared = match_objects(in_objs, out_objs)

        n_in = len(in_objs)
        n_out = len(out_objs)
        n_moved = 0
        n_recolored = 0
        color_map: dict[int, int] = {}
        color_map_consistent = True

        for pair in matched:
            dr = pair.output_obj.centroid[0] - pair.input_obj.centroid[0]
            dc = pair.output_obj.centroid[1] - pair.input_obj.centroid[1]
            if abs(dr) > 1.0 or abs(dc) > 1.0:
                n_moved += 1
            if pair.input_obj.color != pair.output_obj.color:
                n_recolored += 1
                src, dst = pair.input_obj.color, pair.output_obj.color
                if src in color_map and color_map[src] != dst:
                    color_map_consistent = False
                else:
                    color_map[src] = dst

        if n_recolored == 0:
            color_map_consistent = True  # vacuously true

        mean_dr = 0.0
        mean_dc = 0.0
        if matched:
            mean_dr = sum(p.output_obj.centroid[0] - p.input_obj.centroid[0] for p in matched) / len(matched)
            mean_dc = sum(p.output_obj.centroid[1] - p.input_obj.centroid[1] for p in matched) / len(matched)

        count_preserved = 1.0 if n_in == n_out else 0.0

        mean_in_area = sum(o.area for o in in_objs) / max(n_in, 1)
        mean_out_area = sum(o.area for o in out_objs) / max(n_out, 1)
        area_preserved = 1.0 if abs(mean_in_area - mean_out_area) / max(mean_in_area, 1.0) < 0.1 else 0.0

        task_complexity = min((n_moved + n_recolored) / max(n_in, 1), 1.0)

        vec = np.array([
            n_in / MAX_OBJ,
            n_out / MAX_OBJ,
            n_moved / MAX_OBJ,
            n_recolored / MAX_OBJ,
            len(appeared) / MAX_OBJ,
            len(disappeared) / MAX_OBJ,
            mean_dr / MAX_GRID_DIM,
            mean_dc / MAX_GRID_DIM,
            1.0 if color_map_consistent else 0.0,
            count_preserved,
            area_preserved,
            task_complexity,
            l2_weight,
            l2_loss,
        ], dtype=float)
        return [vec]
