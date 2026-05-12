"""Tests for PatternStore."""
import os
import tempfile

import numpy as np
import pytest

from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.storage.pattern_store import PatternStore


def _make_cell(name: str, vec: list[float]) -> Cell:
    return Cell(name=name, embedding=np.array(vec, dtype=np.float32))


# ---------------------------------------------------------------------------
# Task 1: skeleton / save / load
# ---------------------------------------------------------------------------

def test_load_missing_file_returns_empty():
    """load() returns ([], []) when no patterns.npz file exists."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        patterns, weights = store.load()
        assert patterns == []
        assert weights == []


def test_save_load_roundtrip():
    """After save, a fresh PatternStore load restores all data correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        cell = _make_cell("hello", [1.0, 0.0, 0.0])
        store.merge([cell], [2.5], agent_name="word_agent")
        store.save()

        store2 = PatternStore(cache_dir=tmp)
        patterns, weights = store2.load()

        assert len(patterns) == 1
        assert patterns[0].name == "hello"
        assert abs(weights[0] - 2.5) < 1e-5
        np.testing.assert_allclose(
            patterns[0].as_numpy()[:3].astype(np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            atol=1e-5,
        )


def test_clear_resets_in_memory_state():
    """clear() empties in-memory store; a subsequent load() still returns file contents."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        cell = _make_cell("alpha", [0.0, 1.0, 0.0])
        store.merge([cell], [1.0], agent_name="char_agent")
        store.save()

        # clear in-memory
        store.clear()
        # in-memory is empty
        assert store._vectors is None or len(store._vectors) == 0

        # but loading from disk returns the saved pattern
        store2 = PatternStore(cache_dir=tmp)
        patterns, weights = store2.load()
        assert len(patterns) == 1
        assert patterns[0].name == "alpha"


# ---------------------------------------------------------------------------
# Task 2: merge() -- cosine similarity clustering and running average
# ---------------------------------------------------------------------------

def test_merge_idempotent_running_average():
    """Merging the same pattern twice gives running average weight, count=2."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        cell = _make_cell("word", [1.0, 0.0, 0.0])
        store.merge([cell], [2.0], agent_name="word_agent")
        store.merge([cell], [4.0], agent_name="word_agent")
        store.save()

        store2 = PatternStore(cache_dir=tmp)
        patterns, weights = store2.load()
        assert len(patterns) == 1
        assert abs(weights[0] - 3.0) < 1e-5
        assert store2._counts[0] == 2


def test_merge_below_threshold_stays_separate():
    """Two patterns with cosine similarity < 0.85 produce two separate entries."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        # Orthogonal vectors: cosine similarity = 0.0
        a = _make_cell("alpha", [1.0, 0.0, 0.0])
        b = _make_cell("beta",  [0.0, 1.0, 0.0])
        store.merge([a], [1.0], agent_name="word_agent")
        store.merge([b], [1.0], agent_name="word_agent")
        store.save()

        store2 = PatternStore(cache_dir=tmp)
        patterns, weights = store2.load()
        assert len(patterns) == 2


def test_merge_above_threshold_merges():
    """Two vectors with cosine similarity >= 0.85 are treated as the same pattern."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        # Nearly identical vectors: sim ~= 0.9999
        a = _make_cell("run",    [1.0, 0.01, 0.0])
        b = _make_cell("running", [1.0, 0.02, 0.0])
        store.merge([a], [1.0], agent_name="word_agent")
        store.merge([b], [3.0], agent_name="word_agent")
        store.save()

        store2 = PatternStore(cache_dir=tmp)
        patterns, weights = store2.load()
        assert len(patterns) == 1
        # running average: (1.0*1 + 3.0) / 2 = 2.0
        assert abs(weights[0] - 2.0) < 1e-4


def test_agents_field_accumulates():
    """Two different agents contributing to the same pattern both appear in agents field."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        cell = _make_cell("concept", [1.0, 0.0, 0.0])
        store.merge([cell], [1.0], agent_name="word_agent")
        store.merge([cell], [2.0], agent_name="phrase_agent")
        store.save()

        store2 = PatternStore(cache_dir=tmp)
        store2.load()
        agents_field = store2._agents[0]
        assert "word_agent" in agents_field
        assert "phrase_agent" in agents_field


# ---------------------------------------------------------------------------
# Task 3: find_similar()
# ---------------------------------------------------------------------------

def test_find_similar_empty_store_returns_empty():
    """find_similar() returns [] when the store is empty."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.find_similar(query, top_k=5)
        assert results == []


def test_find_similar_top_k_sorted_descending():
    """find_similar() returns top-k results sorted by descending cosine similarity."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)
        # Three vectors; query = [1,0,0]
        # Similarities: a=1.0, b~=0.707, c=0.0
        a = _make_cell("a", [1.0, 0.0, 0.0])
        b = _make_cell("b", [1.0, 1.0, 0.0])
        c = _make_cell("c", [0.0, 1.0, 0.0])
        store.merge([a], [1.0], agent_name="word_agent")
        store.merge([b], [1.0], agent_name="word_agent")
        store.merge([c], [1.0], agent_name="word_agent")
        store.save()

        store2 = PatternStore(cache_dir=tmp)
        store2.load()
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store2.find_similar(query, top_k=3)

        assert len(results) == 3
        sims = [r[0] for r in results]
        # Must be sorted descending
        assert sims[0] >= sims[1] >= sims[2]
        # Top result must be "a"
        assert results[0][1].name == "a"
        assert abs(results[0][0] - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Task 4: integration -- simulated train_sequence x2
# ---------------------------------------------------------------------------

def test_two_train_sequences_weight_grows():
    """Simulated train_sequence x2 with same pattern -> weight accumulates, count=2."""
    with tempfile.TemporaryDirectory() as tmp:
        store = PatternStore(cache_dir=tmp)

        # First training sequence: one agent, one pattern
        cell = _make_cell("dog", [1.0, 0.0, 0.0])
        store.merge([cell], [1.0], agent_name="word_agent")
        store.save()

        # Second training sequence: same pattern, higher weight
        store2 = PatternStore(cache_dir=tmp)
        store2.load()
        store2.merge([cell], [3.0], agent_name="word_agent")
        store2.save()

        # Reload and check final state
        store3 = PatternStore(cache_dir=tmp)
        patterns, weights = store3.load()

        assert len(patterns) == 1
        # Running average over two observations: (1.0 + 3.0) / 2 = 2.0
        assert abs(weights[0] - 2.0) < 1e-4
        assert store3._counts[0] == 2
