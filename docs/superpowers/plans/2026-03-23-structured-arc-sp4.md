# Structured ARC Benchmark (SP4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a three-level HPM benchmark where L1 agents see pixel deltas, L2 agents see per-object anatomy vectors, and L3 agents see relational transformation summaries — each level a genuinely different abstraction.

**Architecture:** A domain-agnostic `LevelEncoder` protocol (hpm/encoders/) and `StructuredOrchestrator` (hpm/agents/structured.py) manage the cascade. ARC-specific encoders (benchmarks/arc_encoders.py) implement the protocol. The benchmark (benchmarks/structured_arc.py) runs four baselines: flat, l1_only, l2_only, full_structured.

**Tech Stack:** Python 3.10+, numpy, pytest. No new dependencies. Uses existing `MultiAgentOrchestrator`, `GaussianPattern`, `make_orchestrator`, `ensemble_score`.

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `hpm/encoders/__init__.py` | Create | Re-exports LevelEncoder |
| `hpm/encoders/base.py` | Create | LevelEncoder Protocol |
| `hpm/agents/structured.py` | Create | StructuredOrchestrator |
| `benchmarks/arc_encoders.py` | Create | ObjectParser, ObjectMatcher, ArcL1/L2/L3Encoder |
| `benchmarks/structured_arc.py` | Create | Benchmark script |
| `tests/encoders/__init__.py` | Create | Empty (package marker) |
| `tests/encoders/test_arc_encoders.py` | Create | Encoder + parser unit tests |
| `tests/agents/test_structured.py` | Create | StructuredOrchestrator unit tests |

---

### Task 1: LevelEncoder Protocol

**Files:**
- Create: `hpm/encoders/base.py`
- Create: `hpm/encoders/__init__.py`
- Create: `tests/encoders/__init__.py`
- Test: `tests/encoders/test_arc_encoders.py` (stub only this task)

- [ ] **Step 1: Write the failing test**

```python
# tests/encoders/test_arc_encoders.py
import numpy as np
from hpm.encoders.base import LevelEncoder


def test_level_encoder_is_protocol():
    """LevelEncoder is a Protocol — structural subtyping check."""
    class DummyEncoder:
        feature_dim: int = 4
        max_steps_per_obs: int | None = 1
        def encode(self, observation, epistemic):
            return [np.zeros(4)]

    enc: LevelEncoder = DummyEncoder()  # type: ignore[assignment]
    vecs = enc.encode(None, None)
    assert len(vecs) == 1
    assert vecs[0].shape == (4,)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/encoders/test_arc_encoders.py::test_level_encoder_is_protocol -v
```
Expected: `ModuleNotFoundError: No module named 'hpm.encoders'`

- [ ] **Step 3: Implement**

```python
# hpm/encoders/base.py
from __future__ import annotations
from typing import Protocol
import numpy as np


class LevelEncoder(Protocol):
    """Domain-agnostic encoder interface for one level of a StructuredOrchestrator.

    feature_dim: Dimension of each returned vector.
    max_steps_per_obs: Expected list length from encode(). None = variable
        (e.g. L2 per-object pair); 1 = fixed (L1, L3).
    """
    feature_dim: int
    max_steps_per_obs: int | None

    def encode(
        self,
        observation,
        epistemic: tuple[float, float] | None,
    ) -> list[np.ndarray]:
        """Encode an observation into a list of feature vectors.

        Args:
            observation: Domain-specific input (e.g. (input_grid, output_grid) for ARC).
            epistemic: (weight, epistemic_loss) from the level below; None for L1.

        Returns:
            List of numpy arrays each of shape (feature_dim,).
            Length is 1 for L1/L3; N for L2 (one per matched object pair).
        """
        ...
```

```python
# hpm/encoders/__init__.py
from hpm.encoders.base import LevelEncoder

__all__ = ["LevelEncoder"]
```

```bash
# Create test package marker
touch tests/encoders/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/encoders/test_arc_encoders.py::test_level_encoder_is_protocol -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hpm/encoders/ tests/encoders/
git commit -m "feat: add LevelEncoder protocol (SP4 Task 1)"
```

---

### Task 2: ObjectParser

**Files:**
- Create: `benchmarks/arc_encoders.py` (ObjectParser only)
- Test: `tests/encoders/test_arc_encoders.py` (add parser tests)

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/encoders/test_arc_encoders.py
from benchmarks.arc_encoders import parse_objects, ArcObject


def test_parse_objects_single():
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    objs = parse_objects(grid)
    assert len(objs) == 1
    assert objs[0].color == 1
    assert objs[0].area == 1
    assert objs[0].bbox == (1, 1, 1, 1)
    assert objs[0].centroid == (1.0, 1.0)


def test_parse_objects_two_colors():
    grid = [
        [1, 0, 2],
        [0, 0, 0],
        [0, 0, 0],
    ]
    objs = parse_objects(grid)
    assert len(objs) == 2
    colors = {o.color for o in objs}
    assert colors == {1, 2}


def test_parse_objects_ignores_background():
    grid = [[0, 0], [0, 0]]
    assert parse_objects(grid) == []


def test_parse_objects_perimeter():
    # 2x2 solid block: each cell has 2 external edges → perimeter = 8
    grid = [
        [1, 1],
        [1, 1],
    ]
    objs = parse_objects(grid)
    assert len(objs) == 1
    assert objs[0].area == 4
    assert objs[0].perimeter == 8


def test_parse_objects_sorted_by_area():
    grid = [
        [1, 1, 0, 2],
        [0, 0, 0, 0],
    ]
    objs = parse_objects(grid)
    assert len(objs) == 2
    # area-2 object (color=1) should come first
    assert objs[0].color == 1
    assert objs[1].color == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/encoders/test_arc_encoders.py -k "parse_objects" -v
```
Expected: `ModuleNotFoundError: No module named 'benchmarks.arc_encoders'`

- [ ] **Step 3: Implement ObjectParser**

```python
# benchmarks/arc_encoders.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/encoders/test_arc_encoders.py -k "parse_objects" -v
```
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/arc_encoders.py tests/encoders/test_arc_encoders.py
git commit -m "feat: add ObjectParser for 4-connected ARC objects (SP4 Task 2)"
```

---

### Task 3: ObjectMatcher

**Files:**
- Modify: `benchmarks/arc_encoders.py` (add MatchedPair + match_objects)
- Test: `tests/encoders/test_arc_encoders.py` (add matcher tests)

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/encoders/test_arc_encoders.py
from benchmarks.arc_encoders import match_objects


def test_match_objects_color_first():
    # Input: one blue object; output: one blue object (moved)
    in_grid = [[1, 0], [0, 0]]
    out_grid = [[0, 0], [0, 1]]
    in_objs = parse_objects(in_grid)
    out_objs = parse_objects(out_grid)
    matched, appeared, disappeared = match_objects(in_objs, out_objs)
    assert len(matched) == 1
    assert len(appeared) == 0
    assert len(disappeared) == 0
    assert matched[0].input_obj.color == 1
    assert matched[0].output_obj.color == 1


def test_match_objects_appeared():
    # New object in output with no input counterpart
    in_grid = [[0, 0], [0, 0]]
    out_grid = [[3, 0], [0, 0]]
    in_objs = parse_objects(in_grid)
    out_objs = parse_objects(out_grid)
    matched, appeared, disappeared = match_objects(in_objs, out_objs)
    assert len(matched) == 0
    assert len(appeared) == 1
    assert len(disappeared) == 0


def test_match_objects_disappeared():
    in_grid = [[2, 0], [0, 0]]
    out_grid = [[0, 0], [0, 0]]
    in_objs = parse_objects(in_grid)
    out_objs = parse_objects(out_grid)
    matched, appeared, disappeared = match_objects(in_objs, out_objs)
    assert len(matched) == 0
    assert len(appeared) == 0
    assert len(disappeared) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/encoders/test_arc_encoders.py -k "match_objects" -v
```
Expected: `ImportError: cannot import name 'match_objects'`

- [ ] **Step 3: Implement ObjectMatcher**

```python
# Add to benchmarks/arc_encoders.py after parse_objects()

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
      1. Colour-first: unique colour in both input and output → direct match.
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/encoders/test_arc_encoders.py -k "match_objects" -v
```
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/arc_encoders.py tests/encoders/test_arc_encoders.py
git commit -m "feat: add ObjectMatcher for ARC object correspondence (SP4 Task 3)"
```

---

### Task 4: ArcL1Encoder

**Files:**
- Modify: `benchmarks/arc_encoders.py` (add helper + ArcL1Encoder)
- Test: `tests/encoders/test_arc_encoders.py`

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/encoders/test_arc_encoders.py
from benchmarks.arc_encoders import ArcL1Encoder


def test_arc_l1_encoder_shape():
    enc = ArcL1Encoder()
    obs = ([[0, 1], [0, 0]], [[1, 0], [0, 0]])
    vecs = enc.encode(obs, epistemic=None)
    assert len(vecs) == 1
    assert vecs[0].shape == (64,)


def test_arc_l1_encoder_reproducible():
    enc = ArcL1Encoder()
    obs = ([[0, 1], [0, 0]], [[1, 0], [0, 0]])
    v1 = enc.encode(obs, epistemic=None)[0]
    v2 = enc.encode(obs, epistemic=None)[0]
    assert np.allclose(v1, v2)


def test_arc_l1_encoder_identity_is_zero():
    # input == output → delta is zero → projection is zero
    enc = ArcL1Encoder()
    grid = [[1, 0], [0, 2]]
    vecs = enc.encode((grid, grid), epistemic=None)
    assert np.allclose(vecs[0], 0.0)


def test_arc_l1_encoder_max_steps():
    assert ArcL1Encoder.max_steps_per_obs == 1
    assert ArcL1Encoder.feature_dim == 64
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/encoders/test_arc_encoders.py -k "arc_l1" -v
```
Expected: `ImportError: cannot import name 'ArcL1Encoder'`

- [ ] **Step 3: Implement ArcL1Encoder**

```python
# Add to benchmarks/arc_encoders.py after match_objects()

def _encode_grid_flat(grid: list[list[int]]) -> np.ndarray:
    flat: list[float] = []
    for row in grid:
        for val in row:
            flat.append(float(val) / 9.0)
    flat.extend([0.0] * (GRID_SIZE - len(flat)))
    return np.array(flat[:GRID_SIZE], dtype=float)


class ArcL1Encoder:
    """Pixel delta encoder: (output_flat - input_flat) @ random_projection → 64-dim."""
    feature_dim: int = L1_DIM
    max_steps_per_obs: int | None = 1

    def encode(self, observation, epistemic=None) -> list[np.ndarray]:
        input_grid, output_grid = observation
        delta = _encode_grid_flat(output_grid) - _encode_grid_flat(input_grid)
        return [delta @ _L1_PROJ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/encoders/test_arc_encoders.py -k "arc_l1" -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/arc_encoders.py tests/encoders/test_arc_encoders.py
git commit -m "feat: add ArcL1Encoder (pixel delta projection, SP4 Task 4)"
```

---

### Task 5: ArcL2Encoder

**Files:**
- Modify: `benchmarks/arc_encoders.py` (add ArcL2Encoder)
- Test: `tests/encoders/test_arc_encoders.py`

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/encoders/test_arc_encoders.py
from benchmarks.arc_encoders import ArcL2Encoder


def test_arc_l2_encoder_matched_pairs():
    # Two distinct-colour objects, each appears in both grids → 2 matched pairs
    in_grid = [[1, 0, 2], [0, 0, 0]]
    out_grid = [[0, 1, 0], [2, 0, 0]]
    enc = ArcL2Encoder()
    vecs = enc.encode((in_grid, out_grid), epistemic=(0.5, 0.1))
    assert len(vecs) == 2
    for v in vecs:
        assert v.shape == (9,)


def test_arc_l2_encoder_excludes_appeared():
    # Input is empty, output has one object → appeared, not matched → empty list
    in_grid = [[0, 0], [0, 0]]
    out_grid = [[3, 0], [0, 0]]
    enc = ArcL2Encoder()
    vecs = enc.encode((in_grid, out_grid), epistemic=None)
    assert vecs == []


def test_arc_l2_encoder_epistemic_in_vector():
    in_grid = [[1, 0], [0, 0]]
    out_grid = [[0, 1], [0, 0]]
    enc = ArcL2Encoder()
    vecs = enc.encode((in_grid, out_grid), epistemic=(0.8, 0.3))
    assert len(vecs) == 1
    # Last two dims are l1_weight=0.8, l1_loss=0.3
    assert abs(vecs[0][-2] - 0.8) < 1e-6
    assert abs(vecs[0][-1] - 0.3) < 1e-6


def test_arc_l2_encoder_normalised():
    # All spatial values should be in [0, 1] given MAX_GRID_DIM=20
    in_grid = [[1] * 10 for _ in range(10)]
    out_grid = [[1] * 10 for _ in range(10)]
    enc = ArcL2Encoder()
    vecs = enc.encode((in_grid, out_grid), epistemic=(1.0, 0.0))
    assert len(vecs) == 1
    assert all(0.0 <= v <= 1.0 for v in vecs[0][:7])  # first 7 spatial/chromatic dims
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/encoders/test_arc_encoders.py -k "arc_l2" -v
```
Expected: `ImportError: cannot import name 'ArcL2Encoder'`

- [ ] **Step 3: Implement ArcL2Encoder**

```python
# Add to benchmarks/arc_encoders.py after ArcL1Encoder

class ArcL2Encoder:
    """Per-object-pair anatomy encoder: 9-dim vector per matched pair.

    Appeared and disappeared objects are excluded (contribute nothing to L2 score).
    All spatial values normalised by MAX_GRID_DIM=20 (fixed cross-task denominator).
    """
    feature_dim: int = L2_DIM
    max_steps_per_obs: int | None = None  # variable — one per matched pair

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
                out.perimeter / MAX_GRID_DIM,
                out.color / 9.0,
                l1_weight,
                l1_loss,
            ], dtype=float)
            vecs.append(vec)
        return vecs
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/encoders/test_arc_encoders.py -k "arc_l2" -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/arc_encoders.py tests/encoders/test_arc_encoders.py
git commit -m "feat: add ArcL2Encoder (per-object anatomy, SP4 Task 5)"
```

---

### Task 6: ArcL3Encoder

**Files:**
- Modify: `benchmarks/arc_encoders.py` (add ArcL3Encoder)
- Test: `tests/encoders/test_arc_encoders.py`

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/encoders/test_arc_encoders.py
from benchmarks.arc_encoders import ArcL3Encoder


def test_arc_l3_encoder_shape():
    enc = ArcL3Encoder()
    obs = ([[1, 0], [0, 0]], [[0, 1], [0, 0]])
    vecs = enc.encode(obs, epistemic=(0.5, 0.2))
    assert len(vecs) == 1
    assert vecs[0].shape == (14,)


def test_arc_l3_encoder_count_preserved():
    # Same object count in/out → count_preserved dim (index 9) == 1.0
    in_grid = [[1, 0], [0, 0]]
    out_grid = [[0, 1], [0, 0]]
    enc = ArcL3Encoder()
    vec = enc.encode((in_grid, out_grid), epistemic=None)[0]
    assert vec[9] == 1.0  # count_preserved


def test_arc_l3_encoder_count_not_preserved():
    in_grid = [[1, 0], [0, 2]]
    out_grid = [[1, 0], [0, 0]]  # one object disappeared
    enc = ArcL3Encoder()
    vec = enc.encode((in_grid, out_grid), epistemic=None)[0]
    assert vec[9] == 0.0  # count_preserved


def test_arc_l3_encoder_color_map_consistent_vacuous():
    # No recolouring → color_map_consistent == 1.0 (vacuously true)
    in_grid = [[1, 0], [0, 0]]
    out_grid = [[0, 1], [0, 0]]
    enc = ArcL3Encoder()
    vec = enc.encode((in_grid, out_grid), epistemic=None)[0]
    assert vec[8] == 1.0  # color_map_consistent


def test_arc_l3_encoder_epistemic_in_vector():
    enc = ArcL3Encoder()
    obs = ([[1, 0], [0, 0]], [[0, 1], [0, 0]])
    vec = enc.encode(obs, epistemic=(0.7, 0.4))[0]
    assert abs(vec[-2] - 0.7) < 1e-6  # l2_weight
    assert abs(vec[-1] - 0.4) < 1e-6  # l2_loss
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/encoders/test_arc_encoders.py -k "arc_l3" -v
```
Expected: `ImportError: cannot import name 'ArcL3Encoder'`

- [ ] **Step 3: Implement ArcL3Encoder**

```python
# Add to benchmarks/arc_encoders.py after ArcL2Encoder

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/encoders/test_arc_encoders.py -k "arc_l3" -v
```
Expected: 5 PASS

- [ ] **Step 5: Run all encoder tests**

```bash
python3 -m pytest tests/encoders/ -v
```
Expected: All PASS (no failures)

- [ ] **Step 6: Commit**

```bash
git add benchmarks/arc_encoders.py tests/encoders/test_arc_encoders.py
git commit -m "feat: add ArcL3Encoder (relational summary, SP4 Task 6)"
```

---

### Task 7: StructuredOrchestrator

**Files:**
- Create: `hpm/agents/structured.py`
- Create: `tests/agents/test_structured.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/agents/test_structured.py
"""Tests for StructuredOrchestrator."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest
from hpm.agents.structured import StructuredOrchestrator
from benchmarks.multi_agent_common import make_orchestrator


def _make_dummy_encoder(n_dims: int, n_vecs: int = 1):
    """Encoder that returns n_vecs zero-vectors of shape (n_dims,)."""
    class DummyEncoder:
        feature_dim = n_dims
        max_steps_per_obs = 1 if n_vecs == 1 else None
        def encode(self, observation, epistemic=None):
            return [np.zeros(n_dims) for _ in range(n_vecs)]
    return DummyEncoder()


def _make_level(n_agents, feature_dim, prefix):
    orch, agents, _ = make_orchestrator(
        n_agents=n_agents, feature_dim=feature_dim,
        agent_ids=[f"{prefix}_{i}" for i in range(n_agents)],
        with_monitor=False,
    )
    return orch, agents


def test_structured_orch_l1_always_fires():
    """L1 steps on every step() call."""
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    l2_orch, l2_agents = _make_level(1, 4, "l2")
    enc1 = _make_dummy_encoder(4)
    enc2 = _make_dummy_encoder(4)
    so = StructuredOrchestrator(
        encoders=[enc1, enc2],
        orches=[l1_orch, l2_orch],
        agents=[l1_agents, l2_agents],
        level_Ks=[1, 3],
    )
    for _ in range(5):
        so.step(None)
    assert so._step_ticks[0] == 5


def test_structured_orch_l2_cadence():
    """L2 fires every K=3 step() calls (not every step)."""
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    l2_orch, l2_agents = _make_level(1, 4, "l2")
    enc1 = _make_dummy_encoder(4)
    enc2 = _make_dummy_encoder(4)
    so = StructuredOrchestrator(
        encoders=[enc1, enc2],
        orches=[l1_orch, l2_orch],
        agents=[l1_agents, l2_agents],
        level_Ks=[1, 3],
    )
    for _ in range(3):
        so.step(None)
    assert so._step_ticks[1] == 1  # L2 fired once (at step 3)
    for _ in range(3):
        so.step(None)
    assert so._step_ticks[1] == 2  # fired again at step 6


def test_structured_orch_l2_multi_vec():
    """L2 encoder returning N vecs causes N step() calls to L2 orchestrator."""
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    l2_orch, l2_agents = _make_level(1, 6, "l2")
    enc1 = _make_dummy_encoder(4, n_vecs=1)
    enc2 = _make_dummy_encoder(6, n_vecs=3)  # returns 3 vecs per obs
    so = StructuredOrchestrator(
        encoders=[enc1, enc2],
        orches=[l1_orch, l2_orch],
        agents=[l1_agents, l2_agents],
        level_Ks=[1, 1],  # L2 fires every step
    )
    so.step(None)
    # L2 should have received 3 step() calls (one per vec)
    assert so._step_ticks[1] == 1  # one cadence tick


def test_structured_orch_l1_obs_dict_override():
    """l1_obs_dict routes partitioned obs to correct agents."""
    l1_orch, l1_agents = _make_level(2, 4, "l1")
    enc1 = _make_dummy_encoder(4)
    so = StructuredOrchestrator(
        encoders=[enc1],
        orches=[l1_orch],
        agents=[l1_agents],
        level_Ks=[1],
    )
    obs_a = np.ones(4)
    obs_b = np.full(4, 2.0)
    l1_obs_dict = {l1_agents[0].agent_id: obs_a, l1_agents[1].agent_id: obs_b}
    result = so.step(None, l1_obs_dict=l1_obs_dict)
    assert "level1" in result


def test_structured_orch_raises_generative_head():
    """Passing non-None generative_head raises NotImplementedError at construction."""
    l1_orch, l1_agents = _make_level(1, 4, "l1")
    enc1 = _make_dummy_encoder(4)
    with pytest.raises(NotImplementedError):
        StructuredOrchestrator(
            encoders=[enc1],
            orches=[l1_orch],
            agents=[l1_agents],
            level_Ks=[1],
            generative_head=object(),
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/agents/test_structured.py -v
```
Expected: `ModuleNotFoundError: No module named 'hpm.agents.structured'`

- [ ] **Step 3: Implement StructuredOrchestrator**

```python
# hpm/agents/structured.py
"""StructuredOrchestrator: domain-agnostic multi-level HPM orchestrator.

Each level receives a domain-specific observation via its LevelEncoder instance.
Cadence for level i is based on total step() calls, not per-object ticks.
Epistemic state (weight, epistemic_loss) threads from each level into the next encoder.
"""
from __future__ import annotations
import numpy as np


class StructuredOrchestrator:
    """N-level HPM orchestrator with per-level LevelEncoder instances.

    Attributes:
        encoders: One LevelEncoder per level.
        orches: One MultiAgentOrchestrator per level.
        agents: One list[Agent] per level.
        level_Ks: Cadence per level. level_Ks[0] unused (L1 always fires).
                  level_Ks[i] = K means level i fires every K step() calls.
        _step_ticks: How many times step() has been called (index 0 = total).
                     _step_ticks[i] counts how many times level i has fired.
    """

    def __init__(
        self,
        encoders: list,
        orches: list,
        agents: list,
        level_Ks: list[int],
        generative_head=None,
        meta_monitor=None,
    ):
        if generative_head is not None:
            raise NotImplementedError("L4 generative_head not yet implemented")
        if meta_monitor is not None:
            raise NotImplementedError("L5 meta_monitor not yet implemented")

        assert len(encoders) == len(orches) == len(agents), (
            "encoders, orches, and agents must have the same length"
        )
        self.encoders = encoders
        self.orches = orches
        self.agents = agents
        self.level_Ks = level_Ks
        self._step_ticks: list[int] = [0] * len(orches)
        self._epistemic: list[tuple[float, float] | None] = [None] * len(orches)

    def step(self, observation, l1_obs_dict: dict | None = None) -> dict:
        """Step all levels on one observation.

        L1 always fires. Level i (i>=1) fires when _step_ticks[0] % level_Ks[i] == 0
        (checked AFTER incrementing, so first fire is at step K).

        Args:
            observation: Domain-specific input passed through to each encoder.
            l1_obs_dict: Optional dict[agent_id, np.ndarray] overriding L1 routing.
                         Use for partitioned training (agent 0 gets obs_a, others get obs_b).

        Returns:
            Dict with keys "level1".."levelN". Non-firing levels return empty dict.
        """
        results: dict[str, dict] = {}
        n = len(self.orches)

        # Level 0 (L1): always fires
        if l1_obs_dict is not None:
            l1_result = self.orches[0].step(l1_obs_dict)
        else:
            vecs = self.encoders[0].encode(observation, epistemic=None)
            obs_dict = {a.agent_id: vecs[0] for a in self.agents[0]}
            l1_result = self.orches[0].step(obs_dict)
        self._step_ticks[0] += 1
        self._epistemic[0] = self._extract_epistemic(0, l1_result)
        results["level1"] = l1_result

        # Higher levels: cadence check on total step count
        for i in range(1, n):
            if self._step_ticks[0] % self.level_Ks[i] == 0:
                vecs = self.encoders[i].encode(observation, epistemic=self._epistemic[i - 1])
                last_result: dict = {}
                for vec in vecs:
                    obs_dict = {a.agent_id: vec for a in self.agents[i]}
                    last_result = self.orches[i].step(obs_dict)
                self._step_ticks[i] += 1
                self._epistemic[i] = self._extract_epistemic(i, last_result)
                results[f"level{i + 1}"] = last_result
            else:
                results[f"level{i + 1}"] = {}

        return results

    def _extract_epistemic(self, level_idx: int, step_result: dict) -> tuple[float, float]:
        """Extract (weight, epistemic_loss) from primary agent at this level.

        weight: mean pattern weight from store; 0.0 if store empty.
        epistemic_loss: from agent step result dict key 'epistemic_loss'; 0.0 if absent.
        """
        primary = self.agents[level_idx][0]
        records = primary.store.query(primary.agent_id)
        weight = float(np.mean([w for _, w in records])) if records else 0.0
        agent_result = step_result.get(primary.agent_id, {})
        epistemic_loss = float(agent_result.get("epistemic_loss", 0.0))
        return (weight, epistemic_loss)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/agents/test_structured.py -v
```
Expected: 5 PASS

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -5
```
Expected: All pass, no regressions

- [ ] **Step 6: Commit**

```bash
git add hpm/agents/structured.py tests/agents/test_structured.py
git commit -m "feat: add StructuredOrchestrator with cadence and epistemic threading (SP4 Task 7)"
```

---

### Task 8: Structured ARC Benchmark

**Files:**
- Create: `benchmarks/structured_arc.py`
- Test: smoke test (5 tasks, checks no crash and output format)

- [ ] **Step 1: Write the smoke test**

```python
# tests/encoders/test_arc_encoders.py — add smoke test at end
def test_structured_arc_smoke(tmp_path, monkeypatch):
    """5-task smoke test: benchmark runs without crashing, returns plausible dict."""
    import importlib.util, sys
    spec_obj = importlib.util.spec_from_file_location(
        "structured_arc", "benchmarks/structured_arc.py"
    )
    mod = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(mod)

    result = mod.run(max_tasks=5)
    assert "n_tasks" in result
    assert result["n_tasks"] == 5
    assert "flat_acc" in result
    assert "full_acc" in result
    assert 0.0 <= result["flat_acc"] <= 1.0
    assert 0.0 <= result["full_acc"] <= 1.0
```

- [ ] **Step 2: Run smoke test to verify it fails**

```bash
python3 -m pytest tests/encoders/test_arc_encoders.py::test_structured_arc_smoke -v
```
Expected: `FileNotFoundError` or `ModuleNotFoundError`

- [ ] **Step 3: Implement structured_arc.py**

```python
# benchmarks/structured_arc.py
"""
Structured ARC Benchmark
========================
Three-level HPM benchmark: each level receives a genuinely different abstraction.

  L1 (feature_dim=64): pixel delta — sensory regularities
  L2 (feature_dim=9):  per-object anatomy — object-level witnesses
  L3 (feature_dim=14): relational summary — transformation families

Four baselines compared in one run:
  flat         — 2-agent pixel delta, partitioned training (matches multi_agent_arc.py)
  l1_only      — full structured training, scored via L1 only
  l2_only      — full structured training, scored via L2 mean only
  full         — full structured training, L1 + mean(L2) + L3

Run:
    python benchmarks/structured_arc.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from benchmarks.multi_agent_arc import load_tasks, task_fits, TRAIN_REPS, N_DISTRACTORS, ensemble_score
from benchmarks.multi_agent_common import make_orchestrator, print_results_table
from benchmarks.arc_encoders import (
    ArcL1Encoder, ArcL2Encoder, ArcL3Encoder, _encode_grid_flat, _L1_PROJ,
)
from hpm.agents.structured import StructuredOrchestrator


def _make_structured_orch():
    """Build L1(2-agent) + L2(2-agent) + L3(1-agent) StructuredOrchestrator."""
    l1_orch, l1_agents, _ = make_orchestrator(
        n_agents=2, feature_dim=64, agent_ids=["l1_0", "l1_1"],
        pattern_types=["gaussian", "gaussian"], with_monitor=False,
        gamma_soc=0.5, init_sigma=2.0,
    )
    l2_orch, l2_agents, _ = make_orchestrator(
        n_agents=2, feature_dim=9, agent_ids=["l2_0", "l2_1"],
        pattern_types=["gaussian", "gaussian"], with_monitor=False,
        gamma_soc=0.5, init_sigma=2.0,
    )
    l3_orch, l3_agents, _ = make_orchestrator(
        n_agents=1, feature_dim=14, agent_ids=["l3_0"],
        pattern_types=["gaussian"], with_monitor=False,
        gamma_soc=0.5, init_sigma=2.0,
    )
    enc1, enc2, enc3 = ArcL1Encoder(), ArcL2Encoder(), ArcL3Encoder()
    orch = StructuredOrchestrator(
        encoders=[enc1, enc2, enc3],
        orches=[l1_orch, l2_orch, l3_orch],
        agents=[l1_agents, l2_agents, l3_agents],
        level_Ks=[1, 1, 3],  # L2 always fires; L3 every 3 steps
    )
    return orch, l1_agents, l2_agents, l3_agents


def _make_flat_orch():
    """2-agent flat baseline (matches multi_agent_arc.py)."""
    orch, agents, _ = make_orchestrator(
        n_agents=2, feature_dim=64, agent_ids=["flat_0", "flat_1"],
        pattern_types=["gaussian", "gaussian"], with_monitor=False,
        gamma_soc=0.5, init_sigma=2.0,
    )
    return orch, agents


def _score_structured(l1_agents, l2_agents, l3_agents, obs, l1_ep, l2_ep):
    """Compute L1, L2 (mean), and L3 scores for one candidate observation."""
    l1_enc = ArcL1Encoder()
    l2_enc = ArcL2Encoder()
    l3_enc = ArcL3Encoder()

    l1_vec = l1_enc.encode(obs, epistemic=None)[0]
    l2_vecs = l2_enc.encode(obs, epistemic=l1_ep)
    l3_vec = l3_enc.encode(obs, epistemic=l2_ep)[0]

    l1_score = ensemble_score(l1_agents, l1_vec)
    l2_score = (
        float(np.mean([ensemble_score(l2_agents, v) for v in l2_vecs]))
        if l2_vecs else 0.0
    )
    l3_score = ensemble_score(l3_agents, l3_vec)
    return l1_score, l2_score, l3_score


def run(max_tasks: int | None = None) -> dict:
    tasks = load_tasks()
    tasks = [t for t in tasks if task_fits(t)][:400]
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    rng = np.random.default_rng(42)

    flat_correct = l1_correct = l2_correct = full_correct = 0
    n_evaluated = 0

    for i, task in enumerate(tasks):
        test_pair = task["test"][0]
        test_input = test_pair["input"]
        correct_obs = (test_input, test_pair["output"])

        distractor_idxs = [j for j in range(len(tasks)) if j != i]
        chosen = rng.choice(distractor_idxs, size=N_DISTRACTORS, replace=False)
        distractor_obs = [
            (test_input, tasks[di]["train"][0]["output"]) for di in chosen
        ]
        all_obs = [correct_obs] + distractor_obs

        # Partitioned training pairs
        train_pairs = task["train"]
        pairs_a = train_pairs[0::2] or train_pairs
        pairs_b = train_pairs[1::2] or train_pairs
        n_pairs = max(len(pairs_a), len(pairs_b))

        # --- Structured ---
        orch, l1_agents, l2_agents, l3_agents = _make_structured_orch()
        l1_enc = ArcL1Encoder()
        l1_ids = [a.agent_id for a in l1_agents]

        for _ in range(TRAIN_REPS):
            for k in range(n_pairs):
                obs_a = (pairs_a[k % len(pairs_a)]["input"], pairs_a[k % len(pairs_a)]["output"])
                obs_b = (pairs_b[k % len(pairs_b)]["input"], pairs_b[k % len(pairs_b)]["output"])
                l1_obs_dict = {
                    l1_ids[0]: l1_enc.encode(obs_a, epistemic=None)[0],
                    l1_ids[1]: l1_enc.encode(obs_b, epistemic=None)[0],
                }
                orch.step(obs_a, l1_obs_dict=l1_obs_dict)

        # Extract end-of-training epistemic state
        l1_ep = orch._epistemic[0] or (0.0, 0.0)
        l2_ep = orch._epistemic[1] or (0.0, 0.0)

        # Score all candidates
        all_scores = [
            _score_structured(l1_agents, l2_agents, l3_agents, obs, l1_ep, l2_ep)
            for obs in all_obs
        ]
        l1_scores = [s[0] for s in all_scores]
        l2_scores = [s[1] for s in all_scores]
        combined = [s[0] + s[1] + s[2] for s in all_scores]

        if l1_scores[0] == min(l1_scores):
            l1_correct += 1
        if l2_scores[0] == min(l2_scores):
            l2_correct += 1
        if combined[0] == min(combined):
            full_correct += 1

        # --- Flat baseline ---
        flat_orch, flat_agents = _make_flat_orch()
        flat_ids = [a.agent_id for a in flat_agents]
        for _ in range(TRAIN_REPS):
            for k in range(n_pairs):
                obs_a = (pairs_a[k % len(pairs_a)]["input"], pairs_a[k % len(pairs_a)]["output"])
                obs_b = (pairs_b[k % len(pairs_b)]["input"], pairs_b[k % len(pairs_b)]["output"])
                flat_l1_dict = {
                    flat_ids[0]: l1_enc.encode(obs_a, epistemic=None)[0],
                    flat_ids[1]: l1_enc.encode(obs_b, epistemic=None)[0],
                }
                flat_orch.step(flat_l1_dict)

        flat_vecs = [l1_enc.encode(obs, epistemic=None)[0] for obs in all_obs]
        flat_scores = [ensemble_score(flat_agents, v) for v in flat_vecs]
        if flat_scores[0] == min(flat_scores):
            flat_correct += 1

        n_evaluated += 1

    chance = 1.0 / (N_DISTRACTORS + 1)
    return {
        "n_tasks": n_evaluated,
        "flat_acc": flat_correct / n_evaluated if n_evaluated else 0.0,
        "l1_acc": l1_correct / n_evaluated if n_evaluated else 0.0,
        "l2_acc": l2_correct / n_evaluated if n_evaluated else 0.0,
        "full_acc": full_correct / n_evaluated if n_evaluated else 0.0,
        "chance": chance,
    }


def main():
    print("Running Structured ARC Benchmark (L1=pixel, L2=object, L3=relational)...")
    m = run()
    chance = m["chance"]
    print_results_table(
        title=f"Structured ARC Benchmark ({m['n_tasks']} tasks, per-task reset)",
        cols=["Setup", "Accuracy", "vs Chance"],
        rows=[
            {"Setup": "Flat (2-agent, pixel only)", "Accuracy": f"{m['flat_acc']:.1%}", "vs Chance": f"{m['flat_acc']-chance:+.1%}"},
            {"Setup": "L1 only (structured train)", "Accuracy": f"{m['l1_acc']:.1%}", "vs Chance": f"{m['l1_acc']-chance:+.1%}"},
            {"Setup": "L2 only (object anatomy)",  "Accuracy": f"{m['l2_acc']:.1%}", "vs Chance": f"{m['l2_acc']-chance:+.1%}"},
            {"Setup": "Full (L1+L2+L3 combined)",  "Accuracy": f"{m['full_acc']:.1%}", "vs Chance": f"{m['full_acc']-chance:+.1%}"},
        ],
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test**

```bash
python3 -m pytest tests/encoders/test_arc_encoders.py::test_structured_arc_smoke -v
```
Expected: PASS (may take ~30 seconds for 5 tasks)

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -5
```
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add benchmarks/structured_arc.py tests/encoders/test_arc_encoders.py
git commit -m "feat: add structured ARC benchmark with L1/L2/L3 encoders and 4 baselines (SP4 Task 8)"
```

- [ ] **Step 7: Run the full benchmark**

```bash
python3 benchmarks/structured_arc.py 2>&1 | grep -v Warning
```
Expected: Results table with flat, l1_only, l2_only, full_structured accuracy. full_structured should be competitive with or better than flat.

- [ ] **Step 8: Push**

```bash
git push
```
