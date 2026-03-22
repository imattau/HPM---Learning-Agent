"""Tests for ArchiveLibrarian."""
import json
import pickle
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hpm.store.contextual_store import SubstrateSignature
from hpm.store.archive_librarian import ArchiveLibrarian, CandidateArchive
from hpm.store.tiered_store import TieredStore
from hpm.patterns.factory import make_pattern


def _make_sig(grid_size=(5, 5), color_count=2, object_count=1, bucket="square"):
    return SubstrateSignature(
        grid_size=grid_size,
        unique_color_count=color_count,
        object_count=object_count,
        aspect_ratio_bucket=bucket,
    )


def _write_index(run_dir: Path, entries: list) -> None:
    (run_dir / "index.json").write_text(json.dumps(entries))


def _make_archive_entry(context_id, archive_path, sig: SubstrateSignature,
                        success=True):
    return {
        "context_id": context_id,
        "archive_path": str(archive_path),
        "signature": {
            "grid_size": list(sig.grid_size),
            "unique_color_count": sig.unique_color_count,
            "object_count": sig.object_count,
            "aspect_ratio_bucket": sig.aspect_ratio_bucket,
        },
        "success_metrics": {"correct": success},
        "timestamp": "2024-01-01T00:00:00",
    }


# ---------------------------------------------------------------------------
# query_archive tests
# ---------------------------------------------------------------------------

def test_query_archive_empty_dir(tmp_path):
    lib = ArchiveLibrarian()
    results = lib.query_archive(_make_sig(), tmp_path / "nonexistent")
    assert results == []


def test_query_archive_returns_matching_candidate(tmp_path):
    sig = _make_sig(grid_size=(5, 5), color_count=2)
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    archive_path = run_dir / "ctx1.pkl"
    archive_path.write_bytes(b"")
    entry = _make_archive_entry("ctx1", archive_path, sig)
    _write_index(run_dir, [entry])

    lib = ArchiveLibrarian()
    results = lib.query_archive(sig, tmp_path)
    assert len(results) == 1
    assert results[0].episode_id == "ctx1"


def test_query_archive_excludes_wrong_grid_size(tmp_path):
    sig_stored = _make_sig(grid_size=(3, 3), color_count=2)
    sig_query = _make_sig(grid_size=(5, 5), color_count=2)
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    archive_path = run_dir / "ctx1.pkl"
    archive_path.write_bytes(b"")
    entry = _make_archive_entry("ctx1", archive_path, sig_stored)
    _write_index(run_dir, [entry])

    lib = ArchiveLibrarian()
    results = lib.query_archive(sig_query, tmp_path)
    assert results == []


def test_query_archive_excludes_color_count_diff_greater_than_1(tmp_path):
    sig_stored = _make_sig(grid_size=(5, 5), color_count=5)
    sig_query = _make_sig(grid_size=(5, 5), color_count=2)
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    archive_path = run_dir / "ctx1.pkl"
    archive_path.write_bytes(b"")
    entry = _make_archive_entry("ctx1", archive_path, sig_stored)
    _write_index(run_dir, [entry])

    lib = ArchiveLibrarian()
    results = lib.query_archive(sig_query, tmp_path)
    assert results == []


def test_query_archive_includes_color_count_diff_of_1(tmp_path):
    sig_stored = _make_sig(grid_size=(5, 5), color_count=3)
    sig_query = _make_sig(grid_size=(5, 5), color_count=2)
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    archive_path = run_dir / "ctx1.pkl"
    archive_path.write_bytes(b"")
    entry = _make_archive_entry("ctx1", archive_path, sig_stored)
    _write_index(run_dir, [entry])

    lib = ArchiveLibrarian()
    results = lib.query_archive(sig_query, tmp_path)
    assert len(results) == 1


def test_query_archive_multiple_runs(tmp_path):
    sig = _make_sig(grid_size=(5, 5), color_count=2)
    for run_name in ["run1", "run2"]:
        run_dir = tmp_path / run_name
        run_dir.mkdir()
        archive_path = run_dir / "ctx.pkl"
        archive_path.write_bytes(b"")
        entry = _make_archive_entry(f"ctx_{run_name}", archive_path, sig)
        _write_index(run_dir, [entry])

    lib = ArchiveLibrarian()
    results = lib.query_archive(sig, tmp_path)
    assert len(results) == 2


def test_query_archive_skips_missing_index(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    # No index.json written

    lib = ArchiveLibrarian()
    results = lib.query_archive(_make_sig(), tmp_path)
    assert results == []


def test_query_archive_candidate_has_correct_fields(tmp_path):
    sig = _make_sig(grid_size=(4, 6), color_count=3, object_count=2, bucket="landscape")
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    archive_path = run_dir / "ctx42.pkl"
    archive_path.write_bytes(b"")
    entry = _make_archive_entry("ctx42", archive_path, sig, success=True)
    _write_index(run_dir, [entry])

    lib = ArchiveLibrarian()
    results = lib.query_archive(sig, tmp_path)
    assert len(results) == 1
    c = results[0]
    assert isinstance(c, CandidateArchive)
    assert c.episode_id == "ctx42"
    assert c.archive_path == archive_path
    assert c.success is True
    assert c.signature.grid_size == (4, 6)


# ---------------------------------------------------------------------------
# run_global_pass tests
# ---------------------------------------------------------------------------

def _make_globals_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS global_patterns (
            id TEXT PRIMARY KEY,
            mu BLOB NOT NULL,
            weight REAL NOT NULL,
            agent_id TEXT NOT NULL,
            is_global INTEGER DEFAULT 0,
            context_ids TEXT DEFAULT '[]'
        )
    """)
    conn.commit()
    conn.close()


def test_run_global_pass_upserts_high_weight(tmp_path):
    db_path = tmp_path / "globals.db"
    _make_globals_db(db_path)
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    tier2 = [(p, 0.8, "agent")]

    lib = ArchiveLibrarian()
    lib.run_global_pass(str(db_path), tier2, "ctx1", weight_threshold=0.6, promotion_n=5)

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT id, context_ids FROM global_patterns WHERE id=?",
                       (p.id,)).fetchone()
    conn.close()
    assert row is not None
    assert "ctx1" in json.loads(row[1])


def test_run_global_pass_skips_low_weight(tmp_path):
    db_path = tmp_path / "globals.db"
    _make_globals_db(db_path)
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    tier2 = [(p, 0.3, "agent")]

    lib = ArchiveLibrarian()
    lib.run_global_pass(str(db_path), tier2, "ctx1", weight_threshold=0.6, promotion_n=5)

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT id FROM global_patterns WHERE id=?", (p.id,)).fetchone()
    conn.close()
    assert row is None


def test_run_global_pass_promotes_after_n_appearances(tmp_path):
    db_path = tmp_path / "globals.db"
    _make_globals_db(db_path)
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    tier2 = [(p, 0.9, "agent")]

    lib = ArchiveLibrarian()
    for i in range(3):
        lib.run_global_pass(str(db_path), tier2, f"ctx{i}", weight_threshold=0.6, promotion_n=3)

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT is_global, context_ids FROM global_patterns WHERE id=?",
                       (p.id,)).fetchone()
    conn.close()
    assert row[0] == 1
    assert len(json.loads(row[1])) >= 3


def test_run_global_pass_deduplicates_context_ids(tmp_path):
    db_path = tmp_path / "globals.db"
    _make_globals_db(db_path)
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    tier2 = [(p, 0.9, "agent")]

    lib = ArchiveLibrarian()
    lib.run_global_pass(str(db_path), tier2, "ctx1", weight_threshold=0.6, promotion_n=5)
    lib.run_global_pass(str(db_path), tier2, "ctx1", weight_threshold=0.6, promotion_n=5)

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT context_ids FROM global_patterns WHERE id=?", (p.id,)).fetchone()
    conn.close()
    ctx_ids = json.loads(row[0])
    assert ctx_ids.count("ctx1") == 1


# ---------------------------------------------------------------------------
# load_global_patterns tests
# ---------------------------------------------------------------------------

def test_load_global_patterns_returns_global_rows(tmp_path):
    db_path = tmp_path / "globals.db"
    _make_globals_db(db_path)
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    mu_blob = pickle.dumps(p.mu)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO global_patterns (id, mu, weight, agent_id, is_global, context_ids) "
        "VALUES (?,?,?,?,1,?)",
        (p.id, mu_blob, 0.9, "agent", json.dumps(["ctx1"]))
    )
    conn.commit()
    conn.close()

    lib = ArchiveLibrarian()
    rows = lib.load_global_patterns(str(db_path))
    assert len(rows) == 1
    assert rows[0][0] == p.id


def test_load_global_patterns_excludes_non_global(tmp_path):
    db_path = tmp_path / "globals.db"
    _make_globals_db(db_path)
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    mu_blob = pickle.dumps(p.mu)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO global_patterns (id, mu, weight, agent_id, is_global, context_ids) "
        "VALUES (?,?,?,?,0,?)",
        (p.id, mu_blob, 0.9, "agent", json.dumps([]))
    )
    conn.commit()
    conn.close()

    lib = ArchiveLibrarian()
    rows = lib.load_global_patterns(str(db_path))
    assert rows == []


def test_load_global_patterns_missing_db(tmp_path):
    lib = ArchiveLibrarian()
    rows = lib.load_global_patterns(str(tmp_path / "nonexistent.db"))
    assert rows == []


def test_load_global_patterns_none_db():
    lib = ArchiveLibrarian()
    rows = lib.load_global_patterns(None)
    assert rows == []
