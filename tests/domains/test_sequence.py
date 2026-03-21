import numpy as np
import pytest
from hpm.domains.sequence import SequenceDomain


def make_domain(vocab_size=6, seed=42):
    return SequenceDomain(vocab_size=vocab_size, seed=seed)


# --- Protocol compliance ---

def test_domain_has_required_methods():
    d = make_domain()
    for method in ('observe', 'feature_dim', 'deep_perturb', 'surface_perturb', 'transfer_probe'):
        assert hasattr(d, method), f"missing method: {method}"


def test_feature_dim():
    d = make_domain(vocab_size=8)
    assert d.feature_dim() == 8


def test_order_not_one_raises():
    with pytest.raises(ValueError, match="order=1"):
        SequenceDomain(order=2)


# --- observe() ---

def test_observe_returns_one_hot():
    d = make_domain()
    x = d.observe()
    assert x.shape == (6,)
    assert x.sum() == pytest.approx(1.0)
    assert set(x.tolist()) <= {0.0, 1.0}


def test_observe_stationary_distribution():
    """Long run should converge to theoretical stationary distribution."""
    d = make_domain(vocab_size=4, seed=0)
    counts = np.zeros(4)
    n = 20_000
    for _ in range(n):
        x = d.observe()
        counts[np.argmax(x)] += 1
    empirical = counts / n
    # Compute theoretical stationary distribution via left eigenvector
    eigvals, eigvecs = np.linalg.eig(d._transition.T)
    stat = np.abs(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))])
    stat /= stat.sum()
    # Allow tolerance for finite sample
    np.testing.assert_allclose(empirical, stat[d._label_map], atol=0.03)


def test_reproducible_with_seed():
    d1 = make_domain(vocab_size=5, seed=7)
    d2 = make_domain(vocab_size=5, seed=7)
    seq1 = [np.argmax(d1.observe()) for _ in range(50)]
    seq2 = [np.argmax(d2.observe()) for _ in range(50)]
    assert seq1 == seq2


# --- deep_perturb() ---

def test_deep_perturb_changes_transition():
    d = make_domain(seed=0)
    d2 = d.deep_perturb()
    assert not np.allclose(d._transition, d2._transition)


def test_deep_perturb_preserves_label_map():
    d = make_domain(seed=0)
    original_label_map = d._label_map.copy()
    d2 = d.deep_perturb()
    np.testing.assert_array_equal(d2._label_map, original_label_map)


def test_deep_perturb_does_not_mutate_rng():
    """Calling deep_perturb() must not advance self._rng."""
    d = make_domain(seed=42)
    d.observe()  # advance one step
    d.deep_perturb()
    obs_after = np.argmax(d.observe())
    d_ref = make_domain(seed=42)
    d_ref.observe()  # advance one step
    obs_ref = np.argmax(d_ref.observe())
    assert obs_after == obs_ref


# --- surface_perturb() ---

def test_surface_perturb_preserves_transition():
    d = make_domain(seed=0)
    original_transition = d._transition.copy()
    d2 = d.surface_perturb()
    np.testing.assert_array_equal(d2._transition, original_transition)


def test_surface_perturb_changes_label_map():
    d = make_domain(seed=0)
    d2 = d.surface_perturb()
    assert not np.array_equal(d._label_map, d2._label_map)


def test_surface_perturb_does_not_mutate_rng():
    """Calling surface_perturb() must not advance self._rng."""
    d = make_domain(seed=42)
    d.observe()  # advance one step
    d.surface_perturb()
    obs_after = np.argmax(d.observe())
    d_ref = make_domain(seed=42)
    d_ref.observe()
    obs_ref = np.argmax(d_ref.observe())
    assert obs_after == obs_ref


# --- transfer_probe() ---

def test_transfer_probe_shapes():
    d = make_domain(vocab_size=5)
    for near in (True, False):
        probe = d.transfer_probe(near=near)
        assert len(probe) == 200
        for x, label in probe:
            assert x.shape == (5,)
            assert 0 <= label < 5


def test_transfer_probe_near_uses_source_label_map():
    """near=True probe must use self._label_map (identical surface to training domain)."""
    d = make_domain(seed=0)
    probe = d.transfer_probe(near=True)
    for x, label in probe:
        observed_idx = np.argmax(x)
        assert observed_idx == d._label_map[label]


def test_transfer_probe_far_differs_from_source_label_map():
    """near=False probe label_map must differ from self._label_map."""
    d = make_domain(seed=0, vocab_size=8)
    probe_near = d.transfer_probe(near=True)
    probe_far = d.transfer_probe(near=False)
    near_indices = [np.argmax(x) for x, _ in probe_near]
    far_indices = [np.argmax(x) for x, _ in probe_far]
    assert near_indices != far_indices


def test_transfer_probe_does_not_mutate_state():
    """transfer_probe() must not change _current or advance _rng."""
    d = make_domain(seed=42)
    d.observe()  # advance one step
    d.transfer_probe(near=True)
    d.transfer_probe(near=False)
    obs_after = np.argmax(d.observe())
    d_ref = make_domain(seed=42)
    d_ref.observe()
    obs_ref = np.argmax(d_ref.observe())
    assert obs_after == obs_ref
