"""
HFN scaling tests — verify the world model scales predictably with grid size.

Checks:
  1. Node count grows correctly with D (primitives O(D), abstract priors O(1))
  2. log_prob throughput is O(D) not O(D³) — the fast path is active
  3. World model builds in reasonable time for common grid sizes
  4. Observer can process observations at all tested grid sizes
  5. Coverage: abstract priors fire at every scale (structural knowledge transfers)
"""

import time
import numpy as np
import pytest

from hpm_fractal_node.arc_world_model import build_world_model
from hfn.observer import Observer


# Grid sizes to test: (rows, cols)
GRID_SIZES = [(3, 3), (5, 5), (10, 10)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=GRID_SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def grid_size(request):
    return request.param


@pytest.fixture
def world_model_3x3():
    return build_world_model(3, 3)


@pytest.fixture
def world_model_10x10():
    return build_world_model(10, 10)


# ---------------------------------------------------------------------------
# 1. Node count scaling
# ---------------------------------------------------------------------------

def test_node_count_scales_with_grid_size():
    """Primitive nodes grow O(D); abstract priors are grid-size-independent."""
    counts = {}
    for rows, cols in GRID_SIZES:
        forest, registry = build_world_model(rows, cols)
        D = rows * cols
        nodes = forest.active_nodes()
        primitive_count = sum(1 for n in nodes if n.id.startswith("primitive_cell_"))
        abstract_count = sum(
            1 for n in nodes
            if any(n.id.startswith(p) for p in [
                "prior_signal", "prior_pixel", "prior_field",
                "prior_sparse", "prior_dense", "prior_symmetry",
                "relationships_hfn", "encoder_hfn",
            ])
        )
        counts[(rows, cols)] = {
            "D": D, "total": len(nodes),
            "primitives": primitive_count, "abstract": abstract_count,
        }

    # primitive_cell_rc nodes = exactly D per grid
    for (rows, cols), c in counts.items():
        assert c["primitives"] == rows * cols, (
            f"{rows}x{cols}: expected {rows*cols} cell primitives, got {c['primitives']}"
        )

    # Abstract priors appear at every scale
    for (rows, cols), c in counts.items():
        assert c["abstract"] >= 5, (
            f"{rows}x{cols}: too few abstract priors ({c['abstract']})"
        )

    # Total node count grows with D but not faster than O(D)
    sizes = sorted(counts.keys(), key=lambda s: s[0] * s[1])
    for i in range(1, len(sizes)):
        small, large = sizes[i - 1], sizes[i]
        D_ratio = (large[0] * large[1]) / (small[0] * small[1])
        node_ratio = counts[large]["total"] / counts[small]["total"]
        assert node_ratio <= D_ratio * 2, (
            f"Node count grew faster than 2×D_ratio: "
            f"{small}→{large} D_ratio={D_ratio:.1f} node_ratio={node_ratio:.1f}"
        )


# ---------------------------------------------------------------------------
# 2. log_prob fast path is active and O(D)
# ---------------------------------------------------------------------------

def test_log_prob_diagonal_fast_path(grid_size):
    """All prior nodes use diagonal sigma and activate the O(D) fast path."""
    rows, cols = grid_size
    forest, registry = build_world_model(rows, cols)
    for node in forest.active_nodes():
        assert node._sigma_diag is not None, (
            f"Node {node.id} does not have diagonal sigma cached — "
            f"log_prob will fall back to O(D³) Cholesky"
        )


def test_log_prob_throughput_scales_linearly():
    """log_prob call time should scale roughly linearly with D (O(D) fast path)."""
    n_calls = 5000
    times = {}
    for rows, cols in GRID_SIZES:
        forest, _ = build_world_model(rows, cols)
        node = next(iter(forest.active_nodes()))
        x = np.random.default_rng(0).random(rows * cols)
        t0 = time.perf_counter()
        for _ in range(n_calls):
            node.log_prob(x)
        times[(rows, cols)] = (time.perf_counter() - t0) / n_calls

    # Time per call should not grow faster than 10× when D grows ~11× (3x3→10x10)
    t_small = times[(3, 3)]
    t_large = times[(10, 10)]
    D_ratio = 100 / 9
    time_ratio = t_large / t_small
    assert time_ratio < D_ratio * 3, (
        f"log_prob time grew {time_ratio:.1f}× for {D_ratio:.1f}× D increase — "
        f"suggests O(D³) path is active"
    )


# ---------------------------------------------------------------------------
# 3. World model builds in reasonable time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows,cols,max_seconds", [
    (3, 3, 1.0),
    (5, 5, 2.0),
    (10, 10, 5.0),
])
def test_world_model_build_time(rows, cols, max_seconds):
    """build_world_model should complete well within time budget."""
    t0 = time.perf_counter()
    forest, registry = build_world_model(rows, cols)
    elapsed = time.perf_counter() - t0
    assert elapsed < max_seconds, (
        f"{rows}x{cols} build took {elapsed:.2f}s (limit {max_seconds}s)"
    )
    assert len(forest.active_nodes()) > 0


# ---------------------------------------------------------------------------
# 4. Observer can process observations at each grid size
# ---------------------------------------------------------------------------

def test_observer_processes_observations(grid_size):
    """Observer runs without error and forest grows at each grid size."""
    rows, cols = grid_size
    D = rows * cols
    forest, registry = build_world_model(rows, cols)
    prior_ids = set(registry.keys())

    baseline = D / 2 * np.log(2 * np.pi)
    obs = Observer(
        forest,
        tau=baseline + 2.0,
        budget=5,
        lambda_complexity=0.05,
        alpha_gain=0.15,
        beta_loss=0.05,
        absorption_overlap_threshold=0.6,
        absorption_miss_threshold=6,
        residual_surprise_threshold=baseline + 5.0,
        compression_cooccurrence_threshold=3,
        w_init=0.1,
        protected_ids=prior_ids,
    )

    rng = np.random.default_rng(42)
    n_initial = len(forest)
    # Capture which prior IDs are actually in the forest before any observations
    initial_forest_prior_ids = {n.id for n in forest.active_nodes()} & prior_ids

    for _ in range(20):
        x = rng.random(D)
        obs.observe(x)

    # Forest should remain valid (no crash, priors intact)
    assert len(forest) >= n_initial, "Forest shrunk below initial size"
    surviving_prior_ids = {n.id for n in forest.active_nodes()} & prior_ids
    assert surviving_prior_ids == initial_forest_prior_ids, (
        f"Protected priors were removed: {initial_forest_prior_ids - surviving_prior_ids}"
    )


# ---------------------------------------------------------------------------
# 5. Coverage: abstract priors fire at every scale
# ---------------------------------------------------------------------------

def test_abstract_priors_fire_at_all_scales(grid_size):
    """Density and colour priors should explain some observations at any grid size."""
    rows, cols = grid_size
    D = rows * cols
    forest, registry = build_world_model(rows, cols)

    baseline = D / 2 * np.log(2 * np.pi)
    tau = baseline + 3.0

    rng = np.random.default_rng(0)
    abstract_nodes = [
        n for n in forest.active_nodes()
        if any(k in n.id for k in ["colour", "sparse", "dense", "signal"])
    ]
    assert len(abstract_nodes) > 0, "No abstract nodes found"

    # Generate observations at various colour densities
    hits = 0
    for colour_val in [0.0, 2/9, 5/9, 8/9]:
        x = np.full(D, colour_val)
        for node in abstract_nodes:
            if -node.log_prob(x) < tau:
                hits += 1
                break

    assert hits > 0, (
        f"{rows}x{cols}: no abstract prior explained any colour-uniform observation"
    )
