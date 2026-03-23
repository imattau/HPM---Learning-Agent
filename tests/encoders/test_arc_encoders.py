"""Unit tests for ARC encoders: ObjectParser, ObjectMatcher, ArcL1/L2/L3Encoder."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest
from hpm.encoders.base import LevelEncoder


def test_level_encoder_is_protocol():
    """LevelEncoder is a Protocol -- structural subtyping check."""
    class DummyEncoder:
        feature_dim: int = 4
        max_steps_per_obs: int | None = 1
        def encode(self, observation, epistemic):
            return [np.zeros(4)]

    enc: LevelEncoder = DummyEncoder()  # type: ignore[assignment]
    vecs = enc.encode(None, None)
    assert len(vecs) == 1
    assert vecs[0].shape == (4,)


# ---------------------------------------------------------------------------
# ObjectParser tests
# ---------------------------------------------------------------------------
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
    # 2x2 solid block: each cell has 2 external edges -> perimeter = 8
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


# ---------------------------------------------------------------------------
# ObjectMatcher tests
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ArcL1Encoder tests
# ---------------------------------------------------------------------------
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
    # input == output -> delta is zero -> projection is zero
    enc = ArcL1Encoder()
    grid = [[1, 0], [0, 2]]
    vecs = enc.encode((grid, grid), epistemic=None)
    assert np.allclose(vecs[0], 0.0)


def test_arc_l1_encoder_max_steps():
    assert ArcL1Encoder.max_steps_per_obs == 1
    assert ArcL1Encoder.feature_dim == 64


# ---------------------------------------------------------------------------
# ArcL2Encoder tests
# ---------------------------------------------------------------------------
from benchmarks.arc_encoders import ArcL2Encoder


def test_arc_l2_encoder_matched_pairs():
    # Two distinct-colour objects, each appears in both grids -> 2 matched pairs
    in_grid = [[1, 0, 2], [0, 0, 0]]
    out_grid = [[0, 1, 0], [2, 0, 0]]
    enc = ArcL2Encoder()
    vecs = enc.encode((in_grid, out_grid), epistemic=(0.5, 0.1))
    assert len(vecs) == 2
    for v in vecs:
        assert v.shape == (9,)


def test_arc_l2_encoder_excludes_appeared():
    # Input is empty, output has one object -> appeared, not matched -> empty list
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


# ---------------------------------------------------------------------------
# ArcL3Encoder tests
# ---------------------------------------------------------------------------
from benchmarks.arc_encoders import ArcL3Encoder


def test_arc_l3_encoder_shape():
    enc = ArcL3Encoder()
    obs = ([[1, 0], [0, 0]], [[0, 1], [0, 0]])
    vecs = enc.encode(obs, epistemic=(0.5, 0.2))
    assert len(vecs) == 1
    assert vecs[0].shape == (14,)


def test_arc_l3_encoder_count_preserved():
    # Same object count in/out -> count_preserved dim (index 9) == 1.0
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
    # No recolouring -> color_map_consistent == 1.0 (vacuously true)
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


# ---------------------------------------------------------------------------
# Structured ARC benchmark smoke test
# ---------------------------------------------------------------------------
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
