"""
ARC Object Encoder — SP31 Object-Level HFN Architecture.

Decomposes ARC grids into connected components and encodes them as
420D object-level vectors:
  [0-199]   Input objects  (K=10, 10×20D each)
  [200-399] Output objects (K=10, 10×20D each)
  [400-419] Rule summary   (20D)

Object descriptor (20D per object):
  [0]    color / 9
  [1]    row_center / 29
  [2]    col_center / 29
  [3]    height / 30
  [4]    width / 30
  [5]    area / 900
  [6-15] shape fingerprint (10 binary features)
  [16-19] reserved / 0
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# Manifold layout constants
K = 10          # Max objects per grid
D_OBJ = 20     # Dims per object descriptor
OBJ_DIM = K * D_OBJ   # 200D per grid
RULE_DIM = 20          # Rule summary
TOTAL_DIM = OBJ_DIM + OBJ_DIM + RULE_DIM   # 420D

IN_SLICE  = slice(0, OBJ_DIM)
OUT_SLICE = slice(OBJ_DIM, 2 * OBJ_DIM)
RULE_SLICE = slice(2 * OBJ_DIM, TOTAL_DIM)


def find_connected_components(grid: np.ndarray, connectivity: int = 4) -> list[tuple[int, np.ndarray]]:
    """
    Extract connected components from grid.

    Args:
        grid: np.ndarray (H, W) with color values 0-9
        connectivity: 4 (default) or 8

    Returns:
        list of (color: int, pixel_coords: np.ndarray(N, 2))
        sorted by (row_center, col_center) for canonical ordering.
    """
    if grid.size == 0:
        return []

    struct = ndimage.generate_binary_structure(2, 1) if connectivity == 4 else ndimage.generate_binary_structure(2, 2)

    components = []
    colors = np.unique(grid)
    for color in colors:
        if color == 0:
            continue
        mask = (grid == color).astype(int)
        labeled, n_comp = ndimage.label(mask, structure=struct)
        for comp_id in range(1, n_comp + 1):
            coords = np.argwhere(labeled == comp_id)
            if coords.size > 0:
                components.append((int(color), coords))

    # Sort by (row_center, col_center) for canonical ordering
    def sort_key(item):
        _, coords = item
        return (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

    components.sort(key=sort_key)
    return components


def compute_object_descriptor(color: int, pixel_coords: np.ndarray, grid_shape: tuple) -> np.ndarray:
    """
    Convert a connected component to a 20D object descriptor.

    Args:
        color: int (0-9)
        pixel_coords: np.ndarray(N, 2) of (row, col) indices
        grid_shape: (H, W) of the source grid

    Returns:
        np.ndarray(20,) descriptor
    """
    desc = np.zeros(D_OBJ)
    H, W = grid_shape

    rows = pixel_coords[:, 0]
    cols = pixel_coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    height = r1 - r0 + 1
    width  = c1 - c0 + 1
    area   = len(pixel_coords)
    row_center = float(rows.mean())
    col_center = float(cols.mean())

    # Basic geometric features
    desc[0] = color / 9.0
    desc[1] = row_center / max(H - 1, 1)
    desc[2] = col_center / max(W - 1, 1)
    desc[3] = height / 30.0
    desc[4] = width  / 30.0
    desc[5] = area   / 900.0

    # Shape fingerprint (10 binary features) [6-15]
    bbox_area = height * width
    fill_ratio = area / max(bbox_area, 1)

    is_dot      = (area == 1)
    is_line_h   = (height == 1 and width > 1)
    is_line_v   = (width == 1 and height > 1)
    is_square   = (height == width and fill_ratio > 0.9)
    is_rect     = (fill_ratio > 0.9 and not is_square)
    is_filled   = (fill_ratio > 0.85)

    # Ring: hollow rectangle — border pixels only
    if bbox_area >= 9 and height >= 3 and width >= 3:
        border_area = 2 * height + 2 * (width - 2)
        is_ring = (abs(area - border_area) <= 1)
    else:
        is_ring = False

    # L-shape: roughly L-shaped bounding box with partial fill
    is_L = (0.35 < fill_ratio < 0.75 and not is_ring and not is_line_h and not is_line_v)
    # Plus/cross: approximately symmetric cross pattern
    is_plus = False
    if height >= 3 and width >= 3 and not is_filled:
        mid_r = r0 + height // 2
        mid_c = c0 + width  // 2
        cross_pixels = set(
            tuple(p) for p in pixel_coords
            if p[0] == mid_r or p[1] == mid_c
        )
        if len(cross_pixels) / max(area, 1) > 0.7:
            is_plus = True

    is_T = False  # Reserved — complex to detect cheaply

    desc[6]  = float(is_square)
    desc[7]  = float(is_rect)
    desc[8]  = float(is_line_h)
    desc[9]  = float(is_line_v)
    desc[10] = float(is_L)
    desc[11] = float(is_T)
    desc[12] = float(is_plus)
    desc[13] = float(is_dot)
    desc[14] = float(is_filled)
    desc[15] = float(is_ring)
    # [16-19] reserved — left as zero

    return desc


def grid_to_objects(grid: np.ndarray, K: int = 10) -> np.ndarray:
    """
    Extract up to K connected components and return canonical object array.

    Args:
        grid: np.ndarray input grid
        K: max objects (default 10)

    Returns:
        np.ndarray(K, 20) — first N rows are computed descriptors,
        remaining rows are zero-padded. Sorted by (row_center, col_center).
    """
    result = np.zeros((K, D_OBJ))
    components = find_connected_components(grid, connectivity=4)
    for i, (color, coords) in enumerate(components[:K]):
        result[i] = compute_object_descriptor(color, coords, grid.shape)
    return result


def compute_rule_summary(in_objs: np.ndarray, out_objs: np.ndarray) -> np.ndarray:
    """
    Compute 20D rule summary comparing input and output object arrays.

    Args:
        in_objs:  np.ndarray(K, 20)
        out_objs: np.ndarray(K, 20)

    Returns:
        np.ndarray(20,) rule summary:
          [0-8]  color_delta histogram (color transitions 0-8)
          [9]    position_changed fraction
          [10]   size_changed fraction
          [11]   count_changed ratio
          [12-19] reserved / 0
    """
    summary = np.zeros(RULE_DIM)

    in_active  = np.any(in_objs  != 0, axis=1)
    out_active = np.any(out_objs != 0, axis=1)
    n_in  = int(np.sum(in_active))
    n_out = int(np.sum(out_active))

    if n_in == 0:
        return summary

    # Color transition histogram: for each input object, find closest output by position
    pos_changed = 0
    size_changed = 0
    matched = 0
    for i in range(n_in):
        if not in_active[i]:
            continue
        in_color   = in_objs[i, 0]   # normalized
        in_row     = in_objs[i, 1]
        in_col     = in_objs[i, 2]
        in_area    = in_objs[i, 5]

        # Find closest output object by position
        best_j = -1
        best_dist = float('inf')
        for j in range(n_out):
            if not out_active[j]:
                continue
            dr = out_objs[j, 1] - in_row
            dc = out_objs[j, 2] - in_col
            dist = dr * dr + dc * dc
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j >= 0:
            matched += 1
            out_color = out_objs[best_j, 0]
            out_row   = out_objs[best_j, 1]
            out_col   = out_objs[best_j, 2]
            out_area  = out_objs[best_j, 5]

            # Color delta histogram (buckets 0-8)
            delta_c = int(round((out_color - in_color) * 9.0))
            bucket  = min(8, max(0, delta_c + 4))
            summary[bucket] += 1.0 / n_in

            # Position changed
            if abs(out_row - in_row) > 0.02 or abs(out_col - in_col) > 0.02:
                pos_changed += 1

            # Size changed
            if abs(out_area - in_area) > 0.002:
                size_changed += 1

    summary[9]  = pos_changed / max(n_in, 1)
    summary[10] = size_changed / max(n_in, 1)
    summary[11] = n_out / max(n_in, 1)
    # [12-19] reserved

    return summary


def task_pair_to_vec(input_grid: np.ndarray, output_grid: np.ndarray, K: int = 10) -> np.ndarray:
    """
    Encode (input, output) pair as 420D object-level vector.

    Args:
        input_grid: np.ndarray
        output_grid: np.ndarray
        K: max objects (default 10)

    Returns:
        np.ndarray(420,) = [in_objs(200), out_objs(200), rule_summary(20)]
    """
    in_objs  = grid_to_objects(input_grid,  K=K)
    out_objs = grid_to_objects(output_grid, K=K)
    rule     = compute_rule_summary(in_objs, out_objs)

    vec = np.zeros(TOTAL_DIM)
    vec[IN_SLICE]   = in_objs.flatten()
    vec[OUT_SLICE]  = out_objs.flatten()
    vec[RULE_SLICE] = rule
    return vec


def test_input_to_vec(input_grid: np.ndarray, K: int = 10) -> np.ndarray:
    """
    Encode test input as 420D vector with output objects and rule zeroed.

    Args:
        input_grid: np.ndarray
        K: max objects (default 10)

    Returns:
        np.ndarray(420,) = [in_objs(200), zeros(200), zeros(20)]
    """
    in_objs = grid_to_objects(input_grid, K=K)
    vec = np.zeros(TOTAL_DIM)
    vec[IN_SLICE] = in_objs.flatten()
    return vec


def reconstruct_output_objects(predicted_vec: np.ndarray, K: int = 10) -> np.ndarray:
    """
    Extract predicted output objects from a 420D vector.

    Args:
        predicted_vec: np.ndarray(420,)
        K: max objects

    Returns:
        np.ndarray(K, 20) output object array
    """
    return predicted_vec[OUT_SLICE].reshape(K, D_OBJ)
