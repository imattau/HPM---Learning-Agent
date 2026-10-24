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


def test_l5_calibrates_surprise_thresholds_from_training_residuals():
    monitor = L5MetaMonitor()
    correct = np.array([1.0, 0.0])
    orthogonal = np.array([0.0, 1.0])
    report = monitor.fit_calibration(
        [correct] * 4 + [orthogonal] * 4,
        [correct] * 8,
    )
    assert report["observations"] == 8
    assert monitor.calibrated
    assert monitor.theta_low == pytest.approx(0.0)
    assert monitor.theta_high == pytest.approx(0.7)
    calibrated_confidence = monitor.strategic_confidence()
    assert calibrated_confidence == pytest.approx(1.0)

    next_round = monitor.fit_calibration([correct], [correct])
    assert next_round["observations"] == 1
    assert next_round["cumulative_observations"] == 9
    round_confidence = monitor.strategic_confidence()

    monitor.update(orthogonal, correct)
    assert monitor.strategic_confidence() < round_confidence
    monitor.reset()
    assert monitor.calibrated  # learned calibration persists across task boundaries
    assert monitor.strategic_confidence() == 1.0  # calibration residuals are not task history

    restored = L5MetaMonitor()
    restored.load_state_dict(monitor.state_dict())
    assert restored.calibrated
    assert restored.theta_low == pytest.approx(monitor.theta_low)
    assert restored.theta_high == pytest.approx(monitor.theta_high)


def test_l5_windowed_surprise_tracks_mid_task_drift():
    monitor = L5MetaMonitor(surprise_ewma_alpha=0.5)
    correct = np.array([1.0, 0.0])
    orthogonal = np.array([0.0, 1.0])
    monitor.update(correct, correct)
    stable = monitor.strategic_confidence()
    for _ in range(4):
        monitor.update(orthogonal, correct)
    assert monitor.strategic_confidence() < stable
    assert monitor._surprise_ewma == pytest.approx(0.65625)
    monitor.reset()
    assert monitor._surprise_ewma is None
    assert not monitor._surprises

    bad_monitor = L5MetaMonitor()
    bad_monitor.fit_calibration([orthogonal] * 4, [correct] * 4)
    assert bad_monitor.strategic_confidence() == pytest.approx(1.0)


def test_l5_calibration_rejects_mismatched_or_empty_training_pairs():
    monitor = L5MetaMonitor()
    with pytest.raises(ValueError, match="equal length"):
        monitor.fit_calibration([np.ones(2)], [])
    with pytest.raises(ValueError, match="at least one"):
        monitor.fit_calibration([], [])


def test_l5_rejects_invalid_prequential_residuals_and_corrupt_state():
    monitor = L5MetaMonitor()
    with pytest.raises(ValueError, match="matching shapes"):
        monitor.fit_calibration([np.ones(2)], [np.ones(3)])
    with pytest.raises(ValueError, match="finite"):
        monitor.fit_calibration([np.array([np.nan])], [np.ones(1)])

    monitor.fit_calibration([np.ones(2)], [np.ones(2)])
    state = monitor.state_dict()
    state["calibration_surprises"] = [float("nan")]
    with pytest.raises(ValueError, match="finite and nonnegative"):
        L5MetaMonitor().load_state_dict(state)


def test_l5_rejects_calibrated_state_without_calibration_history():
    monitor = L5MetaMonitor()
    state = monitor.state_dict()
    state["calibrated"] = True
    with pytest.raises(ValueError, match="include calibration residuals"):
        L5MetaMonitor().load_state_dict(state)
