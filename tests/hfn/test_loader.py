import numpy as np
import pytest

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.loader import HFNLoader, LoadItem


class _TinyLoader(HFNLoader):
    namespace = "test"

    @property
    def dim(self) -> int:
        return 3

    def build(self) -> list[HFN] | list[LoadItem]:
        mu = np.zeros(self.dim)
        sigma = np.ones(self.dim)
        return [self._item(self._make_node("a", mu, sigma), role="prior", protected=True)]


def test_namespace_prefix():
    loader = _TinyLoader()
    items = loader.build_items()
    assert items[0].node.id == "test:a"


def test_load_into_observer_registers_state_and_protection():
    loader = _TinyLoader()
    forest = Forest(D=3)
    obs = Observer(forest, tau=1.0)
    items = loader.load_into(forest, observer=obs)

    nid = items[0].node.id
    assert forest.get(nid) is not None
    assert obs.meta_forest.get(f"state:{nid}") is not None
    assert nid in obs.protected_ids


def test_load_into_forest_sets_protected():
    loader = _TinyLoader()
    forest = Forest(D=3)
    loader.load_into(forest)
    assert "test:a" in forest._protected_ids


def test_validate_duplicate_ids_raises():
    class BadLoader(HFNLoader):
        @property
        def dim(self) -> int:
            return 2

        def build(self) -> list[HFN]:
            mu = np.zeros(self.dim)
            sigma = np.ones(self.dim)
            return [
                HFN(mu=mu, sigma=sigma, id="dup", use_diag=True),
                HFN(mu=mu, sigma=sigma, id="dup", use_diag=True),
            ]

    loader = BadLoader()
    with pytest.raises(ValueError, match="duplicate node ids"):
        loader.load_into(Forest(D=2))


def test_validate_dim_mismatch_raises():
    class BadLoader(HFNLoader):
        @property
        def dim(self) -> int:
            return 3

        def build(self) -> list[HFN]:
            return [HFN(mu=np.zeros(2), sigma=np.ones(2), id="bad", use_diag=True)]

    loader = BadLoader()
    with pytest.raises(ValueError, match="mu shape mismatch"):
        loader.load_into(Forest(D=3))


def test_validate_missing_child_raises():
    class BadLoader(HFNLoader):
        @property
        def dim(self) -> int:
            return 2

        def build(self) -> list[HFN]:
            mu = np.zeros(self.dim)
            sigma = np.ones(self.dim)
            parent = HFN(mu=mu, sigma=sigma, id="parent", use_diag=True)
            missing_child = HFN(mu=mu, sigma=sigma, id="child", use_diag=True)
            parent.add_child(missing_child)
            return [parent]

    loader = BadLoader()
    with pytest.raises(ValueError, match="missing child"):
        loader.load_into(Forest(D=2))


def test_validate_edge_references_non_child_raises():
    class BadLoader(HFNLoader):
        @property
        def dim(self) -> int:
            return 2

        def build(self) -> list[HFN]:
            mu = np.zeros(self.dim)
            sigma = np.ones(self.dim)
            parent = HFN(mu=mu, sigma=sigma, id="parent", use_diag=True)
            child_a = HFN(mu=mu, sigma=sigma, id="a", use_diag=True)
            child_b = HFN(mu=mu, sigma=sigma, id="b", use_diag=True)
            child_c = HFN(mu=mu, sigma=sigma, id="c", use_diag=True)
            parent.add_child(child_a)
            parent.add_child(child_b)
            parent.add_edge(child_c, child_b, "bad_edge")
            return [parent, child_a, child_b, child_c]

    loader = BadLoader()
    with pytest.raises(ValueError, match="edge references non-child"):
        loader.load_into(Forest(D=2))


def test_legacy_build_returns_hfns_normalized():
    class LegacyLoader(HFNLoader):
        @property
        def dim(self) -> int:
            return 2

        def build(self) -> list[HFN]:
            mu = np.zeros(self.dim)
            sigma = np.ones(self.dim)
            return [HFN(mu=mu, sigma=sigma, id="legacy", use_diag=True)]

    loader = LegacyLoader()
    items = loader.load_into(Forest(D=2))
    assert len(items) == 1
    assert isinstance(items[0], LoadItem)
    assert items[0].node.id == "legacy"
