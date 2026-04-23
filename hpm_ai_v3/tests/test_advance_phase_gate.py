"""Test that ADVANCE_PHASE directive is blocked when accuracy < 0.3."""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


def make_meta():
    from hpm_ai_v3.meta_cognitive_pattern import MetaCognitivePattern
    return MetaCognitivePattern()


def make_agent(success_history):
    agent = MagicMock()
    agent._meta_success_history = success_history
    agent._steps_since_advance = 10
    return agent


def make_curriculum():
    curriculum = MagicMock()
    curriculum.active_pattern_idx = 0
    curriculum.patterns = [MagicMock(), MagicMock(), MagicMock()]
    return curriculum


def test_advance_phase_blocked_at_zero_success():
    """ADVANCE_PHASE must not advance curriculum when success rate is 0.0."""
    meta = make_meta()
    agent = make_agent([0.0] * 20)
    curriculum = make_curriculum()

    meta._do_advance_phase(agent, curriculum)

    curriculum.advance_phase.assert_not_called()


def test_advance_phase_blocked_at_low_success():
    """ADVANCE_PHASE must not advance when success rate is below 0.3."""
    meta = make_meta()
    agent = make_agent([0.1, 0.2, 0.0, 0.1, 0.2])  # mean = 0.12
    curriculum = make_curriculum()

    meta._do_advance_phase(agent, curriculum)

    curriculum.advance_phase.assert_not_called()


def test_advance_phase_blocked_with_empty_history():
    """ADVANCE_PHASE must not advance when no history exists."""
    meta = make_meta()
    agent = make_agent([])
    curriculum = make_curriculum()

    meta._do_advance_phase(agent, curriculum)

    curriculum.advance_phase.assert_not_called()


def test_advance_phase_allowed_above_threshold():
    """ADVANCE_PHASE must advance curriculum when success rate >= 0.3."""
    meta = make_meta()
    agent = make_agent([1.0] * 10 + [0.0] * 5)  # mean = 0.667
    curriculum = make_curriculum()

    meta._do_advance_phase(agent, curriculum)

    curriculum.advance_phase.assert_called_once()


def test_advance_phase_allowed_at_exact_threshold():
    """ADVANCE_PHASE must advance at exactly 0.3 success rate."""
    meta = make_meta()
    # 3 successes out of 10 = 0.3 exactly
    agent = make_agent([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    curriculum = make_curriculum()

    meta._do_advance_phase(agent, curriculum)

    curriculum.advance_phase.assert_called_once()


def test_advance_phase_resets_steps_counter_when_allowed():
    """When ADVANCE_PHASE fires, agent._steps_since_advance is reset to 0."""
    meta = make_meta()
    agent = make_agent([1.0] * 20)
    agent._steps_since_advance = 50
    curriculum = make_curriculum()

    meta._do_advance_phase(agent, curriculum)

    assert agent._steps_since_advance == 0
