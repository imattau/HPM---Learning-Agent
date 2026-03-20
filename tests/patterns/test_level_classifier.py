import pytest
from unittest.mock import MagicMock
from hpm.patterns.classifier import HPMLevelClassifier


def make_pattern(conn, comp):
    """Minimal mock pattern with controllable connectivity and compress."""
    p = MagicMock()
    p.connectivity.return_value = conn
    p.compress.return_value = comp
    return p


def test_level_1_when_all_metrics_low():
    clf = HPMLevelClassifier()
    p = make_pattern(conn=0.1, comp=0.1)
    assert clf.compute_level(p, density=0.1) == 1


def test_level_2_from_connectivity_only():
    clf = HPMLevelClassifier()
    p = make_pattern(conn=0.35, comp=0.0)
    assert clf.compute_level(p, density=0.1) == 2


def test_level_3_requires_both_conn_and_comp():
    clf = HPMLevelClassifier()
    p = make_pattern(conn=0.55, comp=0.45)
    assert clf.compute_level(p, density=0.1) == 3


def test_level_3_not_reached_by_conn_alone():
    clf = HPMLevelClassifier()
    p = make_pattern(conn=0.55, comp=0.1)   # comp below L3 threshold
    assert clf.compute_level(p, density=0.1) == 2


def test_level_4_thresholds():
    clf = HPMLevelClassifier()
    p = make_pattern(conn=0.75, comp=0.65)
    assert clf.compute_level(p, density=0.1) == 4


def test_level_5_requires_high_density_plus_structural():
    clf = HPMLevelClassifier()
    p = make_pattern(conn=0.85, comp=0.75)
    assert clf.compute_level(p, density=0.90) == 5


def test_level_5_not_reached_by_density_alone():
    clf = HPMLevelClassifier()
    p = make_pattern(conn=0.1, comp=0.1)
    assert clf.compute_level(p, density=0.95) != 5


def test_level_4_not_upgraded_to_5_without_density():
    clf = HPMLevelClassifier()
    # conn and comp meet L5 structural thresholds but density is low
    p = make_pattern(conn=0.85, comp=0.75)
    assert clf.compute_level(p, density=0.5) == 4


def test_custom_thresholds_honoured():
    clf = HPMLevelClassifier(l2_conn=0.5)   # raise L2 bar
    p = make_pattern(conn=0.35, comp=0.0)   # would be L2 with defaults
    assert clf.compute_level(p, density=0.0) == 1


def test_boundary_values_stay_at_lower_level():
    clf = HPMLevelClassifier()
    # Exactly on the L2 threshold — strict > means stays at L1
    p = make_pattern(conn=0.30, comp=0.0)
    assert clf.compute_level(p, density=0.0) == 1
