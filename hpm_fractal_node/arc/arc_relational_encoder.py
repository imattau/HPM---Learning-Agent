"""
ARC Relational Delta Encoder.

Encodes each (input, output) training pair as an 80D relational delta vector:
- How do objects transform from input to output?
- Translation-invariant: same rule → same vector regardless of object positions.

Feature space (80D):
  [0-69]  Per-object delta slots (K=10, 7D each):
           slot i = [Δrow, Δcol, Δcolor/9, Δsize_ratio, same_shape, is_deleted, is_new]
           sorted: matched pairs first, deleted, then new
  [70-79] Aggregate rule statistics:
           [70] count_delta / K
           [71] frac_moved (|Δrow|>0.03 or |Δcol|>0.03)
           [72] frac_recolored (|Δcolor|>0.5)
           [73] frac_same_shape
           [74] frac_deleted
           [75] frac_new
           [76] mean_Δrow (of matched pairs)
           [77] mean_Δcol (of matched pairs)
           [78] mean_Δcolor/9 (of matched pairs)
           [79] in_count / K
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage

K = 10       # max objects per grid
D_SLOT = 7   # dims per object slot
D_AGG = 10   # aggregate dims
RD_DIM = K * D_SLOT + D_AGG   # 80D total


def find_objects_with_masks(grid: np.ndarray) -> list[dict]:
    """
    Extract connected components from grid with full pixel masks.
    Returns list of dicts with keys: color, mask, row_center, col_center, area
    Sorted by (row_center, col_center) for canonical ordering.
    Background (0) is ignored.
    """
    objects = []
    binary = (grid > 0).astype(int)

    for color in range(1, 10):
        color_mask = (grid == color).astype(int)
        if not color_mask.any():
            continue
        labeled, n = ndimage.label(color_mask)
        for comp_id in range(1, n + 1):
            mask = (labeled == comp_id)
            coords = np.argwhere(mask)
            row_center = float(coords[:, 0].mean())
            col_center = float(coords[:, 1].mean())
            area = int(coords.shape[0])
            objects.append({
                'color': color,
                'mask': mask,
                'row_center': row_center,
                'col_center': col_center,
                'area': area,
            })

    objects.sort(key=lambda o: (o['row_center'], o['col_center']))
    return objects


def match_objects(in_objs: list, out_objs: list) -> tuple[list, list, list]:
    """
    Greedy nearest-neighbor matching of input→output objects.

    Matching cost: Euclidean distance of (row_center, col_center) + color_diff.

    Returns:
      matched_pairs: list of (in_obj, out_obj) tuples
      deleted: list of input objects with no match
      new: list of output objects with no match
    """
    if not in_objs:
        return [], [], list(out_objs)
    if not out_objs:
        return [], list(in_objs), []

    used_out = set()
    matched = []

    for in_obj in in_objs:
        best_j, best_cost = -1, float('inf')
        for j, out_obj in enumerate(out_objs):
            if j in used_out:
                continue
            dr = (in_obj['row_center'] - out_obj['row_center']) ** 2
            dc = (in_obj['col_center'] - out_obj['col_center']) ** 2
            dcolor = abs(in_obj['color'] - out_obj['color'])
            cost = dr + dc + dcolor * 4.0
            if cost < best_cost:
                best_cost = cost
                best_j = j
        if best_j >= 0 and best_cost < 200.0:  # reasonable match threshold
            matched.append((in_obj, out_objs[best_j]))
            used_out.add(best_j)
        else:
            matched.append((in_obj, None))  # deleted

    deleted = [in_obj for in_obj, out_obj in matched if out_obj is None]
    matched_pairs = [(i, o) for i, o in matched if o is not None]
    new_out = [out_objs[j] for j in range(len(out_objs)) if j not in used_out]

    return matched_pairs, deleted, new_out


def compute_relational_delta(input_grid: np.ndarray, output_grid: np.ndarray) -> np.ndarray:
    """
    Encode (input, output) pair as 80D relational delta vector.
    """
    rd = np.zeros(RD_DIM)

    in_objs = find_objects_with_masks(input_grid)
    out_objs = find_objects_with_masks(output_grid)

    H = max(input_grid.shape[0], output_grid.shape[0], 1)
    W = max(input_grid.shape[1], output_grid.shape[1], 1)

    matched_pairs, deleted, new_out = match_objects(in_objs, out_objs)

    slot = 0

    # Matched pairs
    for in_obj, out_obj in matched_pairs:
        if slot >= K:
            break
        base = slot * D_SLOT
        rd[base + 0] = (out_obj['row_center'] - in_obj['row_center']) / H
        rd[base + 1] = (out_obj['col_center'] - in_obj['col_center']) / W
        rd[base + 2] = (out_obj['color'] - in_obj['color']) / 9.0
        in_size = max(in_obj['area'], 1)
        rd[base + 3] = (out_obj['area'] - in_size) / max(in_size, 1)
        # Shape similarity: overlap of masks
        if in_obj['mask'].shape == out_obj['mask'].shape:
            rd[base + 4] = 1.0 if np.array_equal(in_obj['mask'], out_obj['mask']) else 0.0
        else:
            rd[base + 4] = 0.0
        rd[base + 5] = 0.0  # not deleted
        rd[base + 6] = 0.0  # not new
        slot += 1

    # Deleted objects
    for in_obj in deleted:
        if slot >= K:
            break
        base = slot * D_SLOT
        rd[base + 5] = 1.0  # is_deleted
        slot += 1

    # New objects
    for out_obj in new_out:
        if slot >= K:
            break
        base = slot * D_SLOT
        rd[base + 0] = out_obj['row_center'] / H   # absolute position (not delta)
        rd[base + 1] = out_obj['col_center'] / W
        rd[base + 2] = out_obj['color'] / 9.0
        rd[base + 6] = 1.0  # is_new
        slot += 1

    # Aggregate statistics
    n_in = len(in_objs)
    n_out = len(out_objs)
    n_matched = len(matched_pairs)
    n_deleted = len(deleted)
    n_new = len(new_out)

    rd[70] = (n_out - n_in) / K

    if n_matched > 0:
        drows = [(o['row_center'] - i['row_center']) / H for i, o in matched_pairs]
        dcols = [(o['col_center'] - i['col_center']) / W for i, o in matched_pairs]
        dcolors = [(o['color'] - i['color']) / 9.0 for i, o in matched_pairs]

        tol = 0.03
        rd[71] = sum(1 for dr, dc in zip(drows, dcols) if abs(dr) > tol or abs(dc) > tol) / n_matched
        rd[72] = sum(1 for dc in dcolors if abs(dc) > 0.05) / n_matched
        rd[73] = sum(1 for i_o, o_o in matched_pairs
                     if i_o['mask'].shape == o_o['mask'].shape and np.array_equal(i_o['mask'], o_o['mask'])) / n_matched
        rd[76] = float(np.mean(drows))
        rd[77] = float(np.mean(dcols))
        rd[78] = float(np.mean(dcolors))

    rd[74] = n_deleted / max(n_in, 1)
    rd[75] = n_new / max(n_out, 1) if n_out > 0 else 0.0
    rd[79] = n_in / K

    return rd


def compute_test_relational(input_grid: np.ndarray) -> np.ndarray:
    """
    Encode a test input (no output known) as 80D vector.
    Object deltas are zero; only input object count/positions in aggregate.
    """
    rd = np.zeros(RD_DIM)
    in_objs = find_objects_with_masks(input_grid)
    rd[79] = len(in_objs) / K
    # Store input object absolute positions so apply_relational_delta can use them
    H, W = input_grid.shape
    for i, obj in enumerate(in_objs[:K]):
        base = i * D_SLOT
        rd[base + 0] = obj['row_center'] / H
        rd[base + 1] = obj['col_center'] / W
        rd[base + 2] = obj['color'] / 9.0
    return rd


def apply_relational_delta(input_grid: np.ndarray, rd_220d: np.ndarray) -> np.ndarray | None:
    """
    Apply a relational delta (80D slice of target) to produce an output grid.

    For matched objects: translate/recolor the input object pixels.
    For new objects: cannot reconstruct exact shape (return None if new objects needed).

    rd_220d is the predicted 80D target vector.
    """
    # Extract just the 80D portion (first 80 dims of whatever we receive)
    rd = rd_220d[:RD_DIM]

    in_objs = find_objects_with_masks(input_grid)
    if not in_objs and rd[75] < 0.5:
        return None  # empty input, no new objects predicted → can't reconstruct

    H, W = input_grid.shape
    output = np.zeros((H, W), dtype=int)

    n_in = len(in_objs)
    n_matched_slots = 0
    has_new = False

    for slot in range(K):
        base = slot * D_SLOT
        is_deleted = rd[base + 5] > 0.5
        is_new = rd[base + 6] > 0.5

        if is_deleted:
            continue
        if is_new:
            has_new = True
            continue

        # Matched pair: apply delta to input object
        if n_matched_slots < n_in:
            in_obj = in_objs[n_matched_slots]
            dr = float(rd[base + 0]) * H
            dc = float(rd[base + 1]) * W
            dcolor = int(round(float(rd[base + 2]) * 9.0))

            new_color = max(0, min(9, in_obj['color'] + dcolor))
            if new_color == 0:
                new_color = in_obj['color']

            # Shift mask by (dr, dc)
            dr_int = int(round(dr))
            dc_int = int(round(dc))

            coords = np.argwhere(in_obj['mask'])
            for r, c in coords:
                nr, nc = r + dr_int, c + dc_int
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = new_color

            n_matched_slots += 1

    if has_new and not np.any(output):
        return None  # predicted new objects but can't reconstruct them

    return output
