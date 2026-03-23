"""Tests for ContextualPatternStore, SubstrateSignature, and extract_signature."""
import json
import pickle
import sqlite3
import tempfile
import pathlib

import numpy as np
import pytest

from hpm.store.contextual_store import SubstrateSignature, extract_signature
from hpm.store.contextual_store import ContextualPatternStore
from hpm.store.tiered_store import TieredStore
from hpm.patterns.factory import make_pattern


# ---------------------------------------------------------------------------
# SubstrateSignature / extract_signature tests (Task 1)
# ---------------------------------------------------------------------------

def make_grid(rows, cols, colors):
    """Helper: grid with specified dimensions and colour values."""
    g = np.zeros((rows, cols), dtype=int)
    for i, c in enumerate(colors):
        g[i % rows, i % cols] = c
    return g


def test_grid_size():
    grid = np.zeros((5, 8), dtype=int)
    sig = extract_signature(grid)
    assert sig.grid_size == (5, 8)


def test_unique_color_count_excludes_background():
    # Background is 0. Colors 1, 2, 3 appear -> unique_color_count = 3
    grid = np.array([[0, 1, 2], [3, 0, 0]], dtype=int)
    sig = extract_signature(grid)
    assert sig.unique_color_count == 3


def test_unique_color_count_all_background():
    grid = np.zeros((4, 4), dtype=int)
    sig = extract_signature(grid)
    assert sig.unique_color_count == 0


def test_object_count_single_blob():
    grid = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=int)
    sig = extract_signature(grid)
    assert sig.object_count == 1


def test_object_count_two_separated_blobs():
    grid = np.array([[1, 0, 2], [0, 0, 0], [0, 0, 0]], dtype=int)
    sig = extract_signature(grid)
    assert sig.object_count == 2


def test_aspect_ratio_square():
    grid = np.zeros((5, 5), dtype=int)
    sig = extract_signature(grid)
    assert sig.aspect_ratio_bucket == "square"


def test_aspect_ratio_landscape():
    # rows/cols < 0.8 -> landscape
    grid = np.zeros((3, 10), dtype=int)
    sig = extract_signature(grid)
    assert sig.aspect_ratio_bucket == "landscape"


def test_aspect_ratio_portrait():
    # rows/cols > 1.25 -> portrait
    grid = np.zeros((10, 3), dtype=int)
    sig = extract_signature(grid)
    assert sig.aspect_ratio_bucket == "portrait"


def test_signature_is_hashable():
    grid = np.zeros((4, 4), dtype=int)
    sig = extract_signature(grid)
    assert hash(sig) is not None  # frozen dataclass


# ---------------------------------------------------------------------------
# ContextualPatternStore archive write tests (Task 2)
# ---------------------------------------------------------------------------

def _make_store_with_pattern(agent_id="agent"):
    """Helper: TieredStore with one pattern in Tier 2."""
    tiered = TieredStore()
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    # Save directly to Tier 2 (no active context)
    tiered.save(p, 0.9, agent_id)
    return tiered, p


def test_end_context_writes_pkl_and_index(tmp_path):
    tiered, p = _make_store_with_pattern()
    store = ContextualPatternStore(tiered, archive_dir=str(tmp_path), run_id="run1")

    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=3,
                              object_count=2, aspect_ratio_bucket="square")
    context_id = store.begin_context(sig, first_obs=[])

    store.end_context(context_id, success_metrics={"correct": True})

    pkl_path = tmp_path / "run1" / f"{context_id}.pkl"
    assert pkl_path.exists(), "archive .pkl file must be written"

    index_path = tmp_path / "run1" / "index.json"
    assert index_path.exists(), "index.json must be written"

    index = json.loads(index_path.read_text())
    assert len(index) == 1
    entry = index[0]
    assert entry["context_id"] == context_id
    assert entry["signature"]["grid_size"] == [5, 5]
    assert "timestamp" in entry


def test_end_context_write_is_atomic(tmp_path):
    """No .tmp file should remain after end_context."""
    tiered = TieredStore()
    store = ContextualPatternStore(tiered, archive_dir=str(tmp_path), run_id="run1")
    sig = SubstrateSignature(grid_size=(3, 3), unique_color_count=1,
                              object_count=1, aspect_ratio_bucket="square")
    context_id = store.begin_context(sig, first_obs=[])
    store.end_context(context_id, success_metrics={})
    tmp_files = list((tmp_path / "run1").glob("*.tmp.pkl"))
    assert tmp_files == [], "no .tmp.pkl files should remain"


# ---------------------------------------------------------------------------
# Warm-start round-trip and integration tests (Task 3)
# ---------------------------------------------------------------------------

def test_round_trip_warm_start(tmp_path):
    """end_context archives Tier 2; begin_context on matching sig restores patterns."""
    agent_id = "agent"
    tiered1 = TieredStore()
    p = make_pattern(mu=np.array([1.0, 2.0, 3.0, 4.0]),
                     scale=np.eye(4), pattern_type="gaussian")
    tiered1.save(p, 0.9, agent_id)

    store1 = ContextualPatternStore(tiered1, archive_dir=str(tmp_path), run_id="run1")
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")
    ctx1 = store1.begin_context(sig, first_obs=[])
    store1.end_context(ctx1, success_metrics={"correct": True})

    # New store, same run_id so index is found
    tiered2 = TieredStore()
    store2 = ContextualPatternStore(tiered2, archive_dir=str(tmp_path), run_id="run1")
    store2.begin_context(sig, first_obs=[])

    tier2_records = tiered2.query_tier2_all()
    pattern_ids = [rec[0].id for rec in tier2_records]
    assert p.id in pattern_ids, "warm-started Tier 2 must contain archived pattern"


def test_coarse_filter_excludes_mismatched_grid_size(tmp_path):
    """Archive from (5,5) grid must not warm-start a (3,3) episode."""
    agent_id = "agent"
    tiered1 = TieredStore()
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    tiered1.save(p, 0.9, agent_id)

    store1 = ContextualPatternStore(tiered1, archive_dir=str(tmp_path), run_id="run1")
    sig_55 = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                                 object_count=1, aspect_ratio_bucket="square")
    ctx1 = store1.begin_context(sig_55, first_obs=[])
    store1.end_context(ctx1, success_metrics={})

    tiered2 = TieredStore()
    store2 = ContextualPatternStore(tiered2, archive_dir=str(tmp_path), run_id="run1")
    sig_33 = SubstrateSignature(grid_size=(3, 3), unique_color_count=2,
                                 object_count=1, aspect_ratio_bucket="square")
    store2.begin_context(sig_33, first_obs=[])
    assert tiered2.query_tier2_all() == [], "mismatched grid_size must not warm-start"


def test_coarse_filter_excludes_color_count_outside_range(tmp_path):
    """Archive with color_count=5 must not match sig with color_count=2 (diff=3 > 1)."""
    agent_id = "agent"
    tiered1 = TieredStore()
    tiered1.save(make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian"),
                 0.9, agent_id)

    store1 = ContextualPatternStore(tiered1, archive_dir=str(tmp_path), run_id="run1")
    sig_c5 = SubstrateSignature(grid_size=(5, 5), unique_color_count=5,
                                 object_count=1, aspect_ratio_bucket="square")
    ctx1 = store1.begin_context(sig_c5, first_obs=[])
    store1.end_context(ctx1, success_metrics={})

    tiered2 = TieredStore()
    store2 = ContextualPatternStore(tiered2, archive_dir=str(tmp_path), run_id="run1")
    sig_c2 = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                                 object_count=1, aspect_ratio_bucket="square")
    store2.begin_context(sig_c2, first_obs=[])
    assert tiered2.query_tier2_all() == [], "color_count diff > 1 must not warm-start"


def test_fine_filter_rejects_above_threshold(tmp_path):
    """Candidate whose mean NLL > fingerprint_nll_threshold is rejected."""
    agent_id = "agent"
    tiered1 = TieredStore()
    # Pattern far from zero-obs: NLL will be very high
    p = make_pattern(mu=np.array([100.0, 100.0, 100.0, 100.0]),
                     scale=np.eye(4) * 0.001, pattern_type="gaussian")
    tiered1.save(p, 0.9, agent_id)

    store1 = ContextualPatternStore(tiered1, archive_dir=str(tmp_path), run_id="run1")
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")
    ctx1 = store1.begin_context(sig, first_obs=[])
    store1.end_context(ctx1, success_metrics={})

    tiered2 = TieredStore()
    # Very low threshold forces rejection
    store2 = ContextualPatternStore(tiered2, archive_dir=str(tmp_path), run_id="run1",
                                     fingerprint_nll_threshold=0.001)
    obs = [np.zeros(4)]
    store2.begin_context(sig, first_obs=obs)
    assert tiered2.query_tier2_all() == [], "high-NLL candidate must be rejected"


def test_integration_agent_warm_start(tmp_path):
    """Full integration: agent learns task A, archives, task B warm-starts from task A."""
    from hpm.agents.agent import Agent
    from hpm.config import AgentConfig

    config = AgentConfig(agent_id="agent", feature_dim=4, gamma_soc=0.0)
    tiered1 = TieredStore()
    agent1 = Agent(config, store=tiered1)
    obs = np.random.default_rng(42).normal(size=4)
    for _ in range(5):
        agent1.step(obs)

    store1 = ContextualPatternStore(tiered1, archive_dir=str(tmp_path), run_id="run1")
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")
    ctx1 = store1.begin_context(sig, first_obs=[])
    # Promote a pattern to Tier 2 so archive is non-empty
    records = tiered1.query("agent")
    if records:
        p, w = records[0]
        tiered1.promote_to_tier2(p, w, "agent")
    store1.end_context(ctx1, success_metrics={"correct": True})

    # Task B: new agent and store, same signature
    tiered2 = TieredStore()
    store2 = ContextualPatternStore(tiered2, archive_dir=str(tmp_path), run_id="run1")
    store2.begin_context(sig, first_obs=[])

    assert len(tiered2.query_tier2_all()) > 0, \
        "after warm-start, Tier 2 must contain patterns from task A"


# ---------------------------------------------------------------------------
# AgentConfig global fields test (Task 6)
# ---------------------------------------------------------------------------

def test_agent_config_global_fields():
    from hpm.config import AgentConfig
    cfg = AgentConfig(agent_id="a", feature_dim=4)
    assert cfg.global_weight_threshold == 0.6
    assert cfg.global_promotion_n == 5
    assert cfg.fingerprint_nll_threshold == 50.0


# ---------------------------------------------------------------------------
# SQLite schema test (Task 7)
# ---------------------------------------------------------------------------

def test_schema_migration_creates_globals_table(tmp_path):
    tiered = TieredStore()
    store = ContextualPatternStore(tiered, archive_dir=str(tmp_path), run_id="run1")
    db_path = tmp_path / "run1" / "globals.db"
    assert db_path.exists(), "globals.db must be created at __init__"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='global_patterns'"
    )
    assert cursor.fetchone() is not None, "global_patterns table must exist"
    cursor = conn.execute("PRAGMA table_info(global_patterns)")
    cols = {row[1] for row in cursor.fetchall()}
    assert {"id", "mu", "weight", "agent_id", "is_global", "context_ids"}.issubset(cols)
    conn.close()


# ---------------------------------------------------------------------------
# Global Pass tests (Task 8)
# ---------------------------------------------------------------------------

def test_global_pass_upserts_high_weight_patterns(tmp_path):
    """Patterns with weight > global_weight_threshold are upserted to globals.db."""
    agent_id = "agent"
    tiered = TieredStore()
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    tiered.save(p, 0.8, agent_id)  # weight 0.8 > default threshold 0.6

    store = ContextualPatternStore(tiered, archive_dir=str(tmp_path), run_id="run1",
                                    global_weight_threshold=0.6)
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")
    ctx = store.begin_context(sig, first_obs=[])
    store.end_context(ctx, success_metrics={})

    conn = sqlite3.connect(str(tmp_path / "run1" / "globals.db"))
    row = conn.execute("SELECT id, context_ids FROM global_patterns WHERE id=?",
                       (p.id,)).fetchone()
    conn.close()
    assert row is not None, "high-weight pattern must be upserted to globals.db"
    context_ids = json.loads(row[1])
    assert ctx in context_ids


def test_global_pass_does_not_upsert_low_weight(tmp_path):
    """Patterns with weight <= global_weight_threshold are not written to globals.db."""
    agent_id = "agent"
    tiered = TieredStore()
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    tiered.save(p, 0.3, agent_id)  # weight 0.3 < threshold 0.6

    store = ContextualPatternStore(tiered, archive_dir=str(tmp_path), run_id="run1",
                                    global_weight_threshold=0.6)
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")
    ctx = store.begin_context(sig, first_obs=[])
    store.end_context(ctx, success_metrics={})

    conn = sqlite3.connect(str(tmp_path / "run1" / "globals.db"))
    row = conn.execute("SELECT id FROM global_patterns WHERE id=?", (p.id,)).fetchone()
    conn.close()
    assert row is None, "low-weight pattern must not be upserted"


def test_global_promotion_after_n_appearances(tmp_path):
    """Pattern set to is_global=1 after appearing in >= global_promotion_n episodes."""
    agent_id = "agent"
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")

    # Share the same db across 3 separate ContextualPatternStore instances
    run_dir = tmp_path / "run1"
    run_dir.mkdir(parents=True)
    shared_db = str(run_dir / "globals.db")

    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")

    for _ in range(3):
        tiered = TieredStore()
        tiered.save(p, 0.9, agent_id)
        local_store = ContextualPatternStore(tiered, archive_dir=str(tmp_path), run_id="run1",
                                              global_weight_threshold=0.6, global_promotion_n=3)
        local_store._db_path = shared_db  # point all instances at the same db
        ctx = local_store.begin_context(sig, first_obs=[])
        local_store.end_context(ctx, success_metrics={})

    conn = sqlite3.connect(shared_db)
    row = conn.execute("SELECT is_global, context_ids FROM global_patterns WHERE id=?",
                       (p.id,)).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == 1, "is_global must be 1 after N appearances"
    assert len(json.loads(row[1])) >= 3


# ---------------------------------------------------------------------------
# Inject globals test (Task 9)
# ---------------------------------------------------------------------------

def test_begin_context_injects_global_patterns(tmp_path):
    """is_global=1 patterns from globals.db are injected into Tier 2 at begin_context."""
    from hpm.patterns.factory import make_pattern, pattern_from_dict

    run_dir = tmp_path / "run1"
    run_dir.mkdir(parents=True)
    db_path = run_dir / "globals.db"

    p = make_pattern(mu=np.array([1.0, 2.0, 3.0, 4.0]), scale=np.eye(4),
                     pattern_type="gaussian")
    import pickle as _pickle
    mu_blob = _pickle.dumps(p.mu)

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS global_patterns (
            id TEXT PRIMARY KEY, mu BLOB NOT NULL, weight REAL NOT NULL,
            agent_id TEXT NOT NULL, is_global INTEGER DEFAULT 0,
            context_ids TEXT DEFAULT '[]'
        )
    """)
    conn.execute(
        "INSERT INTO global_patterns (id, mu, weight, agent_id, is_global, context_ids) "
        "VALUES (?,?,?,?,1,?)",
        (p.id, mu_blob, 0.9, "agent", json.dumps(["ctx_prev"]))
    )
    conn.commit()
    conn.close()

    tiered = TieredStore()
    store = ContextualPatternStore(tiered, archive_dir=str(tmp_path), run_id="run1")
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                              object_count=1, aspect_ratio_bucket="square")
    store.begin_context(sig, first_obs=[])

    tier2_ids = [rec[0].id for rec in tiered.query_tier2_all()]
    assert p.id in tier2_ids, "is_global pattern must be injected into Tier 2"


# ---------------------------------------------------------------------------
# Task 1: Archive format round-trip and backward compat
# ---------------------------------------------------------------------------

def _make_contextual_store(archive_dir, dim=8):
    """Helper: ContextualPatternStore with one Tier 2 pattern in a temp dir."""
    tiered = TieredStore()
    store = ContextualPatternStore(tiered_store=tiered, archive_dir=archive_dir)
    p = make_pattern(mu=np.ones(dim), scale=np.eye(dim), pattern_type="gaussian")
    tiered._tier2.save(p, 1.0, "agent_x")
    return store, tiered


def test_load_archive_old_list_format_returns_empty_l3(tmp_path):
    """Old list-format pkl loads without error; _load_archive returns []."""
    store, tiered = _make_contextual_store(str(tmp_path))
    archive_path = str(tmp_path / "old.pkl")
    tier2 = tiered.query_tier2_all()
    with open(archive_path, "wb") as f:
        pickle.dump(tier2, f)
    l3_bundles = store._load_archive(archive_path)
    assert l3_bundles == []
    # Side effect: Tier 2 was loaded from archive
    assert len(tiered.query_tier2_all()) > 0


def test_load_archive_new_dict_format_returns_l3_bundles(tmp_path):
    """New dict-format pkl: _load_archive returns list of (mu, w, eps) tuples."""
    store, tiered = _make_contextual_store(str(tmp_path))
    archive_path = str(tmp_path / "new.pkl")
    tier2 = tiered.query_tier2_all()
    mu = np.array([1.0, 2.0, 3.0])
    payload = {"tier2": tier2, "l3_bundles": [(mu, 0.9, 0.1)]}
    with open(archive_path, "wb") as f:
        pickle.dump(payload, f)
    l3_bundles = store._load_archive(archive_path)
    assert len(l3_bundles) == 1
    stored_mu, stored_w, stored_eps = l3_bundles[0]
    np.testing.assert_array_almost_equal(stored_mu, mu)
    assert stored_w == pytest.approx(0.9)
    assert stored_eps == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Task 2: l3_agents integration
# ---------------------------------------------------------------------------

def _make_mock_l3_agent(dim=10):
    """Minimal mock agent with an InMemoryStore for injection tests."""
    from hpm.store.memory import InMemoryStore
    from unittest.mock import MagicMock
    agent = MagicMock()
    agent.agent_id = "l3_mock"
    agent.store = InMemoryStore()
    agent.config = MagicMock()
    agent.config.feature_dim = dim
    return agent


def test_inject_l3_seeds_agent_store(tmp_path):
    """After begin_context with a matching L3 archive, L3 agent store contains seeded pattern."""
    import datetime

    dim = 10
    tiered = TieredStore()
    mock_l3 = _make_mock_l3_agent(dim=dim)
    store = ContextualPatternStore(
        tiered_store=tiered,
        archive_dir=str(tmp_path),
        l3_agents=[mock_l3],
    )

    # Write an archive with an L3 bundle
    run_dir = tmp_path / store._run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    mu = np.linspace(0, 1, dim)
    tier2_p = make_pattern(mu=np.ones(8), scale=np.eye(8), pattern_type="gaussian")
    tiered._tier2.save(tier2_p, 1.0, "some_agent")
    tier2_state = tiered.query_tier2_all()
    archive_path = str(run_dir / "test_ctx.pkl")
    payload = {"tier2": tier2_state, "l3_bundles": [(mu, 0.8, 0.2)]}
    with open(archive_path, "wb") as f:
        pickle.dump(payload, f)

    # Write index so librarian can find this archive
    index_path = str(run_dir / "index.json")
    index = [{
        "context_id": "test_ctx",
        "signature": {
            "grid_size": [5, 5],
            "unique_color_count": 2,
            "object_count": 3,
            "aspect_ratio_bucket": "square",
        },
        "success_metrics": {"correct": True},
        "archive_path": archive_path,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }]
    with open(index_path, "w") as f:
        json.dump(index, f)

    # begin_context: should inject the L3 bundle into mock_l3.store
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                             object_count=3, aspect_ratio_bucket="square")
    store.begin_context(sig, first_obs=[np.zeros(8)])

    patterns = mock_l3.store.query_all()
    assert len(patterns) >= 1
    np.testing.assert_array_almost_equal(patterns[0][0].mu, mu)


def test_no_l3_agents_fallback_unchanged(tmp_path):
    """With l3_agents=None, begin_context behaviour is identical to existing code."""
    tiered = TieredStore()
    store = ContextualPatternStore(
        tiered_store=tiered,
        archive_dir=str(tmp_path),
        l3_agents=None,
    )
    sig = SubstrateSignature(grid_size=(5, 5), unique_color_count=2,
                             object_count=1, aspect_ratio_bucket="square")
    ctx_id = store.begin_context(sig, first_obs=[np.zeros(8)])
    assert isinstance(ctx_id, str)
