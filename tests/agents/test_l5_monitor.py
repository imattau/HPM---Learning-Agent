import numpy as np
import pytest
from hpm.agents.l5_monitor import L5MetaMonitor


def test_strategic_confidence_no_data():
    """Returns 1.0 when no surprises accumulated yet."""
    monitor = L5MetaMonitor()
    assert monitor.strategic_confidence() == 1.0


def test_update_skips_if_l4_pred_none():
    """update() is a no-op when l4_pred is None."""
    monitor = L5MetaMonitor()
    monitor.update(None, np.ones(12))
    assert monitor.strategic_confidence() == 1.0


def test_low_surprise_exploit_mode():
    """Mean surprise < 0.2 -> strategic_confidence = 0.9."""
    monitor = L5MetaMonitor()
    # identical vectors -> cos_dist=0, mag_delta=0 -> surprise=0
    vec = np.array([1.0, 0.0, 0.0])
    monitor.update(vec.copy(), vec.copy())
    monitor.update(vec.copy(), vec.copy())
    assert monitor.strategic_confidence() == pytest.approx(0.9)


def test_high_surprise_explore_mode():
    """Mean surprise > 0.5 -> strategic_confidence = 0.3."""
    monitor = L5MetaMonitor()
    # Orthogonal unit vectors -> cos_dist=1.0, mag_delta=0 -> surprise=0.7
    pred = np.array([1.0, 0.0, 0.0])
    actual = np.array([0.0, 1.0, 0.0])
    monitor.update(pred, actual)
    monitor.update(pred, actual)
    assert monitor.strategic_confidence() == pytest.approx(0.3)


def test_neutral_zone_interpolation():
    """0.2 <= S_bar <= 0.5 -> strategic_confidence = 1 - S_bar."""
    monitor = L5MetaMonitor()
    # cos_dist = 1 - dot([1/sqrt2, 1/sqrt2, 0], [1, 0, 0]) = 1 - 1/sqrt2 ≈ 0.293
    # surprise = 0.7 * 0.293 + 0.3 * ... (need to check actual value)
    pred = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    actual = np.array([1.0, 0.0, 0.0])
    monitor.update(pred, actual)
    monitor.update(pred, actual)
    s_bar = monitor._surprises[0]  # both same
    expected = 1.0 - s_bar
    assert monitor.strategic_confidence() == pytest.approx(expected, abs=1e-6)


def test_surprise_magnitude_component():
    """Magnitude delta contributes 0.3 when direction is identical."""
    monitor = L5MetaMonitor()
    pred = np.array([1.0, 0.0, 0.0])
    actual = np.array([2.0, 0.0, 0.0])  # same direction, double magnitude
    monitor.update(pred, actual)
    # cos_dist = 0 (same direction), mag_delta = |1-2|/2 = 0.5
    # surprise = 0.7*0 + 0.3*0.5 = 0.15 -> Exploit -> 0.9
    assert monitor.strategic_confidence() == pytest.approx(0.9)


def test_reset_clears_state():
    monitor = L5MetaMonitor()
    vec = np.array([1.0, 0.0, 0.0])
    monitor.update(vec, np.array([0.0, 1.0, 0.0]))  # high surprise
    monitor.reset()
    assert monitor.strategic_confidence() == 1.0
