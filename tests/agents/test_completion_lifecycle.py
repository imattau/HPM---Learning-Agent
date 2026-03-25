import numpy as np

from hpm.agents.completion import PatternLifecycleTracker
from hpm.patterns.gaussian import GaussianPattern


def _pattern(pattern_id="p1", level=4, source_id="parent-1"):
    pattern = GaussianPattern(np.zeros(2), np.eye(2), id=pattern_id, source_id=source_id)
    pattern.level = level
    return pattern


def test_lifecycle_tracker_consolidates_and_tracks_identity_lineage():
    tracker = PatternLifecycleTracker(
        consolidation_window=2,
        stable_weight_threshold=0.4,
        retire_weight_threshold=0.1,
        absence_window=2,
        decay_rate=0.5,
    )
    pattern = _pattern()

    tracker.observe(pattern, weight=0.8, step=1)
    tracker.observe(pattern, weight=0.8, step=2)

    state = tracker.states[pattern.id]
    identity = tracker.identities[pattern.id]

    assert state.lifecycle_state == "stable"
    assert state.reinforcement_count == 2
    assert identity.parent_ids == ("parent-1",)
    assert identity.layer_origin == 4
    assert identity.last_seen_at == 2


def test_lifecycle_tracker_retires_after_absence_window():
    tracker = PatternLifecycleTracker(
        consolidation_window=2,
        stable_weight_threshold=0.4,
        retire_weight_threshold=0.1,
        absence_window=2,
        decay_rate=0.5,
    )
    pattern = _pattern(pattern_id="p2")

    tracker.observe(pattern, weight=0.8, step=1)
    tracker.finalize(set(), step=4)

    assert tracker.states[pattern.id].lifecycle_state == "retired"


def test_lifecycle_tracker_prefers_explicit_parent_ids():
    tracker = PatternLifecycleTracker()
    pattern = GaussianPattern(np.zeros(2), np.eye(2), id="child")
    pattern.parent_ids = ("p-a", "p-b")
    pattern.level = 5

    tracker.observe(pattern, weight=0.5, step=1)

    identity = tracker.identities[pattern.id]
    assert identity.parent_ids == ("p-a", "p-b")


def test_lifecycle_tracker_counts_absence_and_reactivation():
    tracker = PatternLifecycleTracker(absence_window=2)
    pattern = _pattern(pattern_id="p3")

    tracker.observe(pattern, weight=0.8, step=1)
    tracker.finalize(set(), step=2)
    tracker.finalize(set(), step=3)
    tracker.observe(pattern, weight=0.8, step=4)

    state = tracker.states[pattern.id]
    identity = tracker.identities[pattern.id]

    assert state.absence_count >= 2
    assert state.reactivation_count == 1
    assert identity.lineage_kind == "promoted"
    assert identity.source_step == 1
