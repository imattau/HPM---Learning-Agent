from hpm_ai_v5.experiments.run_multiturn_clarification_benchmark import _score_cases, ClarificationCase


def test_score_cases_distinguishes_feedback_vs_cleared_carry():
    class StubAgent:
        def __init__(self):
            self.carry_context = {}

    agent = StubAgent()

    def predictor(text: str) -> str:
        if "ambiguous" in text:
            agent.carry_context["intent_feedback_route_hint"] = "target"
            return "unknown"
        if "clarify" in text:
            return "target" if agent.carry_context else "wrong"
        return "wrong"

    cases = [ClarificationCase(turn1="ambiguous", turn2="clarify", gold_intent="target")]

    with_feedback = _score_cases(predictor, agent, cases, clear_carry_each_turn2=False)
    agent2 = StubAgent()

    def predictor2(text: str) -> str:
        if "ambiguous" in text:
            agent2.carry_context["intent_feedback_route_hint"] = "target"
            return "unknown"
        if "clarify" in text:
            return "target" if agent2.carry_context else "wrong"
        return "wrong"

    without_feedback = _score_cases(predictor2, agent2, cases, clear_carry_each_turn2=True)

    assert with_feedback == 1.0
    assert without_feedback == 0.0
