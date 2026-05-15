# hpm_ai_v6/tests/test_quiz_agent.py
from hpm_ai_v6.agents.quiz_agent import QuizQuestion

def test_quiz_question_fields():
    q = QuizQuestion(
        id="q1",
        question="What is the capital of France?",
        options=["London", "Paris", "Berlin", "Rome"],
        correct_index=1,
        topic="geography",
        explanation="Paris is the capital and largest city of France.",
        difficulty="easy",
        source="bank",
    )
    assert q.id == "q1"
    assert len(q.options) == 4
    assert q.correct_index == 1
    assert q.source == "bank"
