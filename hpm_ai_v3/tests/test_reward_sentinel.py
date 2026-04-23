"""Test that evaluate_solution never returns -1.0 (the null-solution sentinel)."""
import pytest
from unittest.mock import MagicMock, patch


def make_agent():
    """Build a minimal DiscoveryAgent with a numeric task."""
    from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent as DiscoveryAgent
    agent = DiscoveryAgent.__new__(DiscoveryAgent)
    agent.current_task = {"type": "pretraining", "text": "What is 2+2?", "answer": 4.0}
    agent.population = MagicMock()
    return agent


def test_evaluate_solution_none_returns_negative_half():
    """evaluate_solution(None) must return -0.5, not -1.0."""
    agent = make_agent()
    result = agent.evaluate_solution(None)
    assert result == -0.5, f"Expected -0.5 for None solution, got {result}"


def test_evaluate_solution_no_task_returns_negative_half():
    """evaluate_solution with no current_task must return -0.5."""
    agent = make_agent()
    agent.current_task = None
    result = agent.evaluate_solution(42)
    assert result == -0.5, f"Expected -0.5 when no task set, got {result}"


def test_evaluate_solution_never_returns_negative_one():
    """evaluate_solution must NEVER return -1.0 under any input."""
    agent = make_agent()
    for sol in [None, "", 0, [], {}, "garbage", -999, object()]:
        result = agent.evaluate_solution(sol)
        assert result != -1.0, f"Got -1.0 for solution={sol!r} — sentinel leaked into reward"


def test_evaluate_solution_correct_answer_returns_one():
    """Correct numeric answer returns 1.0."""
    agent = make_agent()
    result = agent.evaluate_solution(4.0)
    assert result == 1.0, f"Expected 1.0 for correct answer, got {result}"


def test_evaluate_solution_wrong_answer_returns_negative_half():
    """Wrong numeric answer returns -0.5."""
    agent = make_agent()
    result = agent.evaluate_solution(99.0)
    assert result == -0.5, f"Expected -0.5 for wrong answer, got {result}"
