"""Tests for LevelBundle.strategic_confidence field (SP6 Task 3)."""
import numpy as np
from hpm.agents.hierarchical import LevelBundle


def test_level_bundle_has_strategic_confidence_default():
    """LevelBundle.strategic_confidence defaults to 1.0 (backward compatible)."""
    bundle = LevelBundle(
        agent_id="a",
        mu=np.zeros(4),
        weight=0.5,
        epistemic_loss=0.1,
    )
    assert bundle.strategic_confidence == 1.0


def test_level_bundle_strategic_confidence_settable():
    bundle = LevelBundle(
        agent_id="a",
        mu=np.zeros(4),
        weight=0.5,
        epistemic_loss=0.1,
        strategic_confidence=0.6,
    )
    assert bundle.strategic_confidence == 0.6
