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
