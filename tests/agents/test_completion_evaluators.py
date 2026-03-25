import numpy as np

from hpm.agents.completion import EvaluatorArbitrator, EvaluatorVector, MetaEvaluatorState
from hpm.agents.agent import Agent
from hpm.config import AgentConfig
from hpm.dynamics.recombination import RecombinationOperator


def test_evaluator_arbitrator_fixed_mode_sums_terms():
    arb = EvaluatorArbitrator(mode="fixed")
    assert arb.aggregate(1.0, 2.0, 3.0, 4.0) == 10.0


def test_evaluator_arbitrator_adaptive_mode_weights_terms():
    arb = EvaluatorArbitrator(mode="adaptive")
    score = arb.aggregate(1.0, 2.0, 3.0, 4.0)
    assert abs(score - (0.4 * 1.0 + 0.3 * 2.0 + 0.2 * 4.0 + 0.1 * 3.0)) < 1e-9


def test_evaluator_arbitrator_updates_weights_from_outcome_signal():
    arb = EvaluatorArbitrator(mode="adaptive", learning_rate=0.5)
    initial_state = arb.state()
    updated = arb.update(signal=1.0, predictive=1.0, coherence=0.0, cost=0.0, horizon=0.0)

    assert isinstance(updated, MetaEvaluatorState)
    assert updated.update_count == 1
    assert updated.last_signal == 1.0
    assert updated.weights != initial_state.weights
    assert updated.weights[0] > initial_state.weights[0]
    assert updated.weights[0] > max(updated.weights[1:])


def test_recombination_operator_seed_is_deterministic():
    cfg = AgentConfig(agent_id="recomb", feature_dim=2, min_recomb_level=4, kappa_max=1.0, N_recomb=1)
    from hpm.patterns.gaussian import GaussianPattern

    a = GaussianPattern(np.zeros(2), np.eye(2), id="a")
    b = GaussianPattern(np.ones(2), np.eye(2), id="b")
    c = GaussianPattern(np.full(2, 2.0), np.eye(2), id="c")
    for p, level in ((a, 4), (b, 4), (c, 4)):
        p.level = level
    patterns = [a, b, c]
    weights = np.array([0.2, 0.3, 0.5])
    obs = [np.zeros(2)]

    op1 = RecombinationOperator(seed=99)
    op2 = RecombinationOperator(seed=99)
    result1 = op1.attempt(patterns, weights, obs, cfg, trigger="time")
    result2 = op2.attempt(patterns, weights, obs, cfg, trigger="time")

    assert result1 is not None and result2 is not None
    assert (result1.parent_a_id, result1.parent_b_id, result1.insight_score) == (result2.parent_a_id, result2.parent_b_id, result2.insight_score)


def test_agent_exposes_evaluator_vector_and_arbitration_mode():
    cfg = AgentConfig(agent_id="eval_agent", feature_dim=2, evaluator_arbitration_mode="adaptive")
    agent = Agent(cfg)
    result = agent.step(np.zeros(2))
    vector = result["evaluator_vector"]

    assert set(vector.keys()) == {"predictive", "coherence", "cost", "horizon", "aggregate", "arbitration_mode"}
    assert vector["arbitration_mode"] == "adaptive"
    assert isinstance(vector["aggregate"], float)
    assert "decision_trace" in result
    assert result["decision_trace"]["trace_id"] == "eval_agent:1"
    assert result["decision_trace"]["signal_source"] == "rolling_outcome"
    assert result["decision_trace"]["selected_parent_ids"] == ()
    assert result["meta_evaluator_state"]["update_count"] == 1
    assert result["meta_evaluator_state"]["mode"] == "adaptive"
    assert result["meta_evaluator_state"]["signal_source"] == "rolling_outcome"


from hpm.agents.completion import FieldConstraint
from hpm.field.field import PatternField


def test_agent_applies_field_constraints_to_scores():
    field = PatternField()
    field.add_constraint(FieldConstraint(constraint_type="penalize_complexity", scope="eval_agent", strength=1.0, source="env", timestamp=1))
    cfg = AgentConfig(agent_id="eval_agent", feature_dim=2, evaluator_arbitration_mode="adaptive")
    agent = Agent(cfg, field=field)
    result = agent.step(np.zeros(2))
    assert result["field_constraint_count"] == 1
    assert result["field_constraint_strength"] == 1.0
    assert result["evaluator_vector"]["arbitration_mode"] == "adaptive"
    assert "meta_evaluator_state" in result
    traces = agent.consume_decision_traces()
    assert len(traces) == 1
    assert traces[0]["signal_source"] == "rolling_outcome"


def test_agent_meta_evaluator_state_updates_across_steps():
    cfg = AgentConfig(agent_id="meta_agent", feature_dim=2, evaluator_arbitration_mode="adaptive", meta_evaluator_learning_rate=0.5)
    agent = Agent(cfg)

    first = agent.step(np.zeros(2), reward=5.0)
    second = agent.step(np.zeros(2), reward=-10.0)

    assert first["meta_evaluator_state"]["update_count"] == 1
    assert first["meta_evaluator_state"]["last_signal"] > 0.0
    assert second["meta_evaluator_state"]["update_count"] == 2
    assert second["meta_evaluator_state"]["last_signal"] < 0.0
    assert first["meta_evaluator_state"]["weights"] != second["meta_evaluator_state"]["weights"]
    assert first["decision_trace_count"] == 1
    assert second["decision_trace_count"] == 2
