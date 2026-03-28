import psutil
import pytest

def test_psutil_importable():
    mem = psutil.virtual_memory()
    assert mem.available > 0


import tempfile
from typing import Optional
import numpy as np
from pathlib import Path
from hfn import HFN, Forest
from hfn.tiered_forest import TieredForest


def _make_node(d: int = 4, node_id: Optional[str] = None) -> HFN:
    node = HFN(mu=np.zeros(d), sigma=np.eye(d))
    if node_id:
        node.id = node_id
    return node


def test_tiered_forest_is_forest():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td)
        assert isinstance(tf, Forest)


def test_register_makes_node_active():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td)
        node = _make_node()
        tf.register(node)
        assert node in tf.active_nodes()


def test_contains_after_register():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td)
        node = _make_node(node_id="abc")
        tf.register(node)
        assert "abc" in tf


def test_len_after_register():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td)
        tf.register(_make_node())
        tf.register(_make_node())
        assert len(tf) == 2


def test_hot_count_and_cold_count_initial():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td)
        tf.register(_make_node())
        assert tf.hot_count() == 1
        assert tf.cold_count() == 0


def test_deregister_removes_node():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td)
        node = _make_node(node_id="xyz")
        tf.register(node)
        tf.deregister("xyz")
        assert "xyz" not in tf
        assert len(tf) == 0


def test_deregister_protected_is_noop():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td,
                          protected_ids={"prot"})
        node = _make_node(node_id="prot")
        tf.register(node)
        tf.deregister("prot")  # should be no-op
        assert "prot" in tf
        assert len(tf) == 1


# --- Task 3: LRU eviction tests ---

def test_lru_eviction_to_cold_when_max_hot_exceeded():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=2)
        n1 = _make_node(node_id="n1")
        n2 = _make_node(node_id="n2")
        n3 = _make_node(node_id="n3")
        tf.register(n1)
        tf.register(n2)
        tf.register(n3)  # triggers LRU eviction of n1
        assert tf.hot_count() == 2
        assert tf.cold_count() == 1
        assert "n1" in tf        # still known via mu_index
        assert "n1" not in tf._hot  # but evicted from hot


def test_all_nodes_in_contains_after_eviction():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=2)
        for i in range(5):
            tf.register(_make_node(node_id=f"n{i}"))
        assert len(tf) == 5
        assert tf.hot_count() == 2
        assert tf.cold_count() == 3


def test_cold_file_exists_after_eviction():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=1)
        n1 = _make_node(node_id="n1")
        n2 = _make_node(node_id="n2")
        tf.register(n1)
        tf.register(n2)
        cold_files = list(Path(td).glob("*.npz"))
        assert len(cold_files) == 1


def test_deregister_cold_node_removes_file():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=1)
        tf.register(_make_node(node_id="n1"))
        tf.register(_make_node(node_id="n2"))
        # n1 is now cold
        tf.deregister("n1")
        assert "n1" not in tf
        assert "n2" in tf._hot  # n2 was most-recently-used, stays hot
        cold_files = list(Path(td).glob("*.npz"))
        assert len(cold_files) == 0  # n1 .npz deleted


# --- Task 4: retrieve() and get() tests ---

def test_retrieve_finds_cold_node():
    """retrieve() should find and promote a cold node if it's in top-k."""
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=1)
        # n1: mu=[1,0,0,0], n2: mu=[0,1,0,0]
        n1 = HFN(mu=np.array([1., 0., 0., 0.]), sigma=np.eye(4), id="n1")
        n2 = HFN(mu=np.array([0., 1., 0., 0.]), sigma=np.eye(4), id="n2")
        tf.register(n1)
        tf.register(n2)  # n1 evicted to cold
        # Query close to n1
        results = tf.retrieve(np.array([0.9, 0., 0., 0.]), k=1)
        assert len(results) == 1
        assert results[0].id == "n1"
        # n1 should be promoted back to hot
        assert "n1" in tf._hot


def test_retrieve_returns_up_to_k_nodes():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=10)
        mus = [np.array([float(i == j) for j in range(4)]) for i in range(4)]
        mus.append(np.array([0.5, 0.5, 0., 0.]))
        for i, mu in enumerate(mus):
            tf.register(HFN(mu=mu, sigma=np.eye(4), id=f"n{i}"))
        results = tf.retrieve(np.zeros(4), k=3)
        assert len(results) == 3


def test_get_loads_cold_node():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=1)
        n1 = _make_node(node_id="n1")
        n2 = _make_node(node_id="n2")
        tf.register(n1)
        tf.register(n2)  # n1 evicted to cold
        loaded = tf.get("n1")
        assert loaded is not None
        assert loaded.id == "n1"
        assert "n1" in tf._hot  # promoted to hot


def test_get_returns_none_for_unknown_node():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td)
        result = tf.get("nonexistent")
        assert result is None


# --- Task 5: sweep tests ---

def test_sweep_evicts_lru_hot_nodes_under_ram_pressure():
    """Simulate low RAM: sweep should evict bottom half of hot nodes."""
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(
            D=4, forest_id="test", cold_dir=td,
            max_hot=10, sweep_every=5, min_free_ram_mb=999_999,  # always triggers
        )
        for i in range(6):
            tf.register(_make_node(node_id=f"n{i}"))

        # Simulate sweep_every observations
        for _ in range(5):
            tf._on_observe()

        # Bottom half (3 nodes) should be evicted to cold
        assert tf.hot_count() == 3
        assert tf.cold_count() == 3
        assert len(tf) == 6  # all still known


def test_sweep_does_not_delete_protected_nodes():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(
            D=4, forest_id="test", cold_dir=td,
            max_hot=10, sweep_every=5, min_free_ram_mb=999_999,
            protected_ids={"prot"},
        )
        tf.register(_make_node(node_id="prot"))
        for i in range(5):
            tf.register(_make_node(node_id=f"n{i}"))

        for _ in range(5):
            tf._on_observe()

        # prot should still exist (may be cold but not deleted)
        assert "prot" in tf


def test_sweep_step2_deletes_unprotected_cold_nodes():
    """Step 2: nodes evicted to cold and not re-promoted are deleted on next sweep."""
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(
            D=4, forest_id="test", cold_dir=td,
            max_hot=2, sweep_every=5, min_free_ram_mb=0,  # no RAM pressure
        )
        # Register 4 nodes: n0,n1 in hot, n2,n3 evicted to cold
        for i in range(4):
            tf.register(_make_node(node_id=f"n{i}"))
        assert tf.cold_count() == 2  # n0,n1 cold; n2,n3 hot

        # Trigger sweep (no RAM pressure, just persistence floor)
        for _ in range(5):
            tf._on_observe()

        # Cold unprotected nodes (n0, n1) should be deleted entirely
        assert "n0" not in tf
        assert "n1" not in tf
        assert len(tf) == 2  # only n2, n3 remain


def test_sweep_step2_keeps_protected_cold_nodes():
    """Protected cold nodes are NOT deleted by persistence floor — evicted only."""
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(
            D=4, forest_id="test", cold_dir=td,
            max_hot=2, sweep_every=5, min_free_ram_mb=0,
            protected_ids={"n0"},
        )
        for i in range(4):
            tf.register(_make_node(node_id=f"n{i}"))

        for _ in range(5):
            tf._on_observe()

        # n0 is protected and cold — must still exist
        assert "n0" in tf
        # n1 is unprotected and cold — deleted
        assert "n1" not in tf


def test_on_observe_increments_counter():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, sweep_every=10)
        assert tf._obs_count == 0
        tf._on_observe()
        assert tf._obs_count == 1


# --- Task 6: Forest.get() test ---

def test_forest_base_get():
    from hfn import Forest, HFN
    f = Forest(D=4, forest_id="f")
    node = _make_node(node_id="abc")
    f.register(node)
    assert f.get("abc") is node
    assert f.get("missing") is None


# --- Task 7: world model accepts forest_cls ---

def test_build_nlp_world_model_accepts_tiered_forest():
    from hfn.tiered_forest import TieredForest
    from hpm_fractal_node.nlp.nlp_world_model import build_nlp_world_model
    with tempfile.TemporaryDirectory() as td:
        forest, prior_ids = build_nlp_world_model(
            forest_cls=TieredForest,
            cold_dir=td,
            max_hot=100,
        )
        assert isinstance(forest, TieredForest)
        assert len(prior_ids) == 38
