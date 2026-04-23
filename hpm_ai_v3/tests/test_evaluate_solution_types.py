"""Test evaluate_solution handles all answer/solution type combinations without crashing."""
import pytest
from unittest.mock import MagicMock


def make_agent(answer):
    from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent as DiscoveryAgent
    agent = DiscoveryAgent.__new__(DiscoveryAgent)
    agent.population = MagicMock()
    agent.current_task = {"type": "pretraining", "text": "test", "answer": answer}
    return agent


# -- List answer tests --

def test_list_answer_string_solution_returns_negative_half():
    """String solution against list answer must return -0.5, not raise ValueError."""
    agent = make_agent([0.25, 0.25, 0.5])
    result = agent.evaluate_solution("Normalize [1, 1, 2]")
    assert result == -0.5, f"Expected -0.5 got {result}"


def test_list_answer_correct_list_solution_returns_one():
    """Correct list solution against list answer returns 1.0."""
    agent = make_agent([0.25, 0.25, 0.5])
    result = agent.evaluate_solution([0.25, 0.25, 0.5])
    assert result == 1.0, f"Expected 1.0 got {result}"


def test_list_answer_wrong_list_solution_returns_negative_half():
    """Wrong list solution against list answer returns -0.5."""
    agent = make_agent([0.25, 0.25, 0.5])
    result = agent.evaluate_solution([0.1, 0.1, 0.8])
    assert result == -0.5, f"Expected -0.5 got {result}"


def test_list_answer_approx_correct_returns_one():
    """List solution within floating point tolerance returns 1.0."""
    agent = make_agent([0.3333333333333333, 0.3333333333333333, 0.3333333333333333])
    result = agent.evaluate_solution([1/3, 1/3, 1/3])
    assert result == 1.0, f"Expected 1.0 got {result}"


# -- Numeric answer tests --

def test_numeric_answer_string_solution_returns_negative_half():
    """String solution against numeric answer must return -0.5, not crash."""
    agent = make_agent(1.0)
    result = agent.evaluate_solution("Entropy of [0.5, 0.5]")
    assert result == -0.5, f"Expected -0.5 got {result}"


def test_numeric_answer_correct_returns_one():
    """Correct numeric string solution returns 1.0."""
    agent = make_agent(1.0)
    result = agent.evaluate_solution(1.0)
    assert result == 1.0, f"Expected 1.0 got {result}"


def test_numeric_answer_int_solution_correct():
    """Integer solution matching int answer returns 1.0."""
    agent = make_agent(2)
    result = agent.evaluate_solution(2)
    assert result == 1.0, f"Expected 1.0 got {result}"


# -- No exceptions ever --

def test_evaluate_solution_never_raises():
    """evaluate_solution must not raise under any input combination."""
    problematic_inputs = [
        "Normalize [1, 1, 2]",
        "could not convert",
        None,
        [],
        {},
        float("nan"),
        float("inf"),
        [None, None],
        {"key": "val"},
    ]
    for answer in [1.0, [0.25, 0.25, 0.5], 0, "text"]:
        agent = make_agent(answer)
        for sol in problematic_inputs:
            try:
                result = agent.evaluate_solution(sol)
                assert isinstance(result, float), f"Non-float returned for sol={sol!r}, answer={answer!r}: {result}"
            except Exception as e:
                pytest.fail(f"evaluate_solution raised {type(e).__name__} for sol={sol!r}, answer={answer!r}: {e}")
