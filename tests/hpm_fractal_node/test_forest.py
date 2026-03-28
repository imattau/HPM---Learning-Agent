"""
Structural tests for Forest-as-HFN.

Forest is an HFN whose children are its registered nodes.
Tests cover: registration, retrieval, HFN interface compliance.
Dynamic behaviour (weights, absorption, creation) is tested in test_observer.py.
"""

import numpy as np
import pytest
from hfn.hfn import HFN
from hfn.forest import Forest


def test_forest_is_hfn():
    assert isinstance(Forest(), HFN)


def test_register_adds_to_children():
    forest = Forest()
    node = HFN(mu=np.array([1.0, 0.0]), sigma=np.eye(2), id="a")
    forest.register(node)
    assert node in forest.children()


def test_register_is_idempotent():
    forest = Forest()
    node = HFN(mu=np.array([1.0, 0.0]), sigma=np.eye(2), id="a")
    forest.register(node)
    forest.register(node)
    assert len(forest) == 1


def test_deregister_removes_from_children():
    forest = Forest()
    node = HFN(mu=np.array([1.0, 0.0]), sigma=np.eye(2), id="a")
    forest.register(node)
    forest.deregister(node.id)
    assert node not in forest.children()


def test_forest_gaussian_syncs_with_population():
    forest = Forest(D=2)
    a = HFN(mu=np.array([2.0, 0.0]), sigma=np.eye(2), id="a")
    b = HFN(mu=np.array([0.0, 2.0]), sigma=np.eye(2), id="b")
    forest.register(a)
    forest.register(b)
    expected_mu = np.array([1.0, 1.0])
    assert np.allclose(forest.mu, expected_mu)


def test_retrieve_returns_nearest_nodes():
    forest = Forest(D=2)
    near = HFN(mu=np.array([1.0, 0.0]), sigma=np.eye(2), id="near")
    far  = HFN(mu=np.array([9.0, 0.0]), sigma=np.eye(2), id="far")
    forest.register(near)
    forest.register(far)
    results = forest.retrieve(np.array([1.1, 0.0]), k=1)
    assert results[0] is near


def test_retrieve_respects_k():
    forest = Forest(D=2)
    for i in range(5):
        forest.register(HFN(mu=np.array([float(i), 0.0]), sigma=np.eye(2), id=f"n{i}"))
    assert len(forest.retrieve(np.array([0.0, 0.0]), k=3)) == 3


def test_retrieve_empty_forest():
    assert Forest().retrieve(np.array([0.0, 0.0])) == []


def test_forest_expand_returns_hfn():
    forest = Forest(D=2)
    forest.register(HFN(mu=np.array([1.0, 0.0]), sigma=np.eye(2), id="a"))
    assert isinstance(forest.expand(0), HFN)
    assert isinstance(forest.expand(1), HFN)


def test_forest_can_be_child_of_another_hfn():
    """A Forest is just an HFN — it can be a child of a higher-order node."""
    f1 = Forest(D=2, forest_id="world_a")
    f2 = Forest(D=2, forest_id="world_b")
    parent = f1.recombine(f2)
    assert f1 in parent.children()
    assert f2 in parent.children()
