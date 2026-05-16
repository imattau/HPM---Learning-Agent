"""Tests for PatternPager with sqlite-vec backend."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.storage.pattern_pager import PatternPager


def _make_cell(name: str, dim: int, vec: list[float], weight: float = 1.0) -> Cell:
    return Cell(name=name, dim=dim, embedding=np.array(vec, dtype=float), weight=weight)


def _make_pager(tmp_path: str, agent: str = "test_agent") -> PatternPager:
    return PatternPager(cache_dir=tmp_path, agent_name=agent)


@pytest.fixture
def tmp(tmp_path):
    return str(tmp_path)


def test_save_and_load(tmp):
    pager = _make_pager(tmp)
    cell = _make_cell("pat1", 1, [0.1, 0.2, 0.3])
    pager.save(cell)
    pager.flush()
    loaded = pager.load("pat1", {})
    assert loaded is not None
    assert loaded.name == "pat1"
    assert np.allclose(loaded.as_numpy(), [0.1, 0.2, 0.3], atol=1e-5)
    pager.close()


def test_load_unknown_returns_none(tmp):
    pager = _make_pager(tmp)
    assert pager.load("nonexistent", {}) is None
    pager.close()


def test_has(tmp):
    pager = _make_pager(tmp)
    cell = _make_cell("p1", 1, [1.0, 0.0])
    pager.save(cell)
    pager.flush()
    assert pager.has("p1") is True
    assert pager.has("p2") is False
    pager.close()


def test_duplicate_upsert(tmp):
    pager = _make_pager(tmp)
    pager.save(_make_cell("p1", 1, [1.0, 0.0], weight=1.0))
    pager.flush()
    pager.save(_make_cell("p1", 1, [1.0, 0.0], weight=9.9))
    pager.flush()
    loaded = pager.load("p1", {})
    assert abs(loaded.weight - 9.9) < 0.01
    payloads = pager.iter_index_payloads()
    names = [p["name"] for p in payloads]
    assert names.count("p1") == 1
    pager.close()


def test_iter_index_payloads(tmp):
    pager = _make_pager(tmp)
    for i in range(3):
        pager.save(_make_cell(f"p{i}", 1, [float(i), float(i + 1)]))
    pager.flush()
    payloads = pager.iter_index_payloads()
    assert len(payloads) == 3
    names = {p["name"] for p in payloads}
    assert names == {"p0", "p1", "p2"}
    assert all("embedding" in p for p in payloads)
    pager.close()


def test_load_nearest(tmp):
    pager = _make_pager(tmp)
    # Three unit vectors; query is closest to p1
    pager.save(_make_cell("p0", 1, [1.0, 0.0, 0.0]))
    pager.save(_make_cell("p1", 1, [0.0, 1.0, 0.0]))
    pager.save(_make_cell("p2", 1, [0.0, 0.0, 1.0]))
    pager.flush()
    query = np.array([0.01, 0.99, 0.01])
    result = pager.load_nearest(query, {}, min_similarity=0.5)
    assert result is not None
    assert result.name == "p1"
    pager.close()


def test_load_nearest_excludes(tmp):
    pager = _make_pager(tmp)
    pager.save(_make_cell("p0", 1, [1.0, 0.0, 0.0]))
    pager.save(_make_cell("p1", 1, [0.99, 0.1, 0.0]))
    pager.flush()
    query = np.array([1.0, 0.0, 0.0])
    result = pager.load_nearest(query, {}, min_similarity=0.5, exclude_names=["p0"])
    assert result is not None
    assert result.name == "p1"
    pager.close()


def test_load_nearest_below_threshold_returns_none(tmp):
    pager = _make_pager(tmp)
    pager.save(_make_cell("p0", 1, [1.0, 0.0, 0.0]))
    pager.flush()
    query = np.array([0.0, 0.0, 1.0])
    result = pager.load_nearest(query, {}, min_similarity=0.99)
    assert result is None
    pager.close()


def test_select_evictions_under_limit(tmp):
    pager = _make_pager(tmp, agent="evict_agent")
    pager.max_active_patterns = 10
    patterns = [_make_cell(f"p{i}", 1, [float(i)]) for i in range(5)]
    weights = [1.0] * 5
    assert pager.select_evictions(patterns, weights) == []
    pager.close()


def test_select_evictions_over_limit(tmp):
    pager = _make_pager(tmp, agent="evict_agent")
    pager.max_active_patterns = 3
    patterns = [_make_cell(f"p{i}", 1, [float(i)]) for i in range(5)]
    weights = [5.0, 1.0, 3.0, 0.5, 2.0]
    evictions = pager.select_evictions(patterns, weights)
    assert len(evictions) >= 2
    # Lowest-weight indices should be evicted first (weights 0.5 and 1.0 → indices 3 and 1)
    assert 3 in evictions
    assert 1 in evictions
    pager.close()


def test_persistence_across_open_close(tmp):
    pager = _make_pager(tmp)
    pager.save(_make_cell("persist1", 1, [0.5, 0.5]))
    pager.flush()
    pager.close()

    pager2 = _make_pager(tmp)
    loaded = pager2.load("persist1", {})
    assert loaded is not None
    assert loaded.name == "persist1"
    pager2.close()
