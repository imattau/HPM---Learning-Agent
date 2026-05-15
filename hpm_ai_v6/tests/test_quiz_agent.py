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

from unittest.mock import MagicMock
from hpm_ai_v6.agents.quiz_agent import QuizAgent, QuizQuestion

def _make_agent():
    return QuizAgent(MagicMock(), MagicMock())

def test_load_bank_easy():
    agent = _make_agent()
    questions = agent._load_bank("easy", 5)
    assert len(questions) == 5
    for q in questions:
        assert isinstance(q, QuizQuestion)
        assert len(q.options) == 4
        assert 0 <= q.correct_index <= 3
        assert q.difficulty == "easy"
        assert q.source == "bank"

def test_load_bank_medium():
    agent = _make_agent()
    questions = agent._load_bank("medium", 10)
    assert len(questions) == 10
    assert all(q.difficulty == "medium" for q in questions)

def test_load_bank_hard():
    agent = _make_agent()
    questions = agent._load_bank("hard", 3)
    assert len(questions) == 3
    assert all(q.source == "bank" for q in questions)

def test_generate_quiz_bank_source():
    agent = _make_agent()
    questions = agent.generate_quiz(n=5, difficulty="easy", source="bank")
    assert len(questions) == 5
    assert all(isinstance(q, QuizQuestion) for q in questions)

def test_generate_question_returns_quiz_question():
    reader = MagicMock()
    reader.top_concepts.return_value = ["photosynthesis", "gravity", "democracy"]

    reasoner = MagicMock()
    reasoner.reason.return_value = (
        '{"question": "What process do plants use to make food?", '
        '"options": ["Respiration", "Photosynthesis", "Fermentation", "Digestion"], '
        '"correct_index": 1, '
        '"explanation": "Photosynthesis converts light energy into glucose."}'
    )

    agent = QuizAgent(reader, reasoner)
    q = agent.generate_question(topic=None, difficulty="easy")

    assert q is not None
    assert isinstance(q, QuizQuestion)
    assert len(q.options) == 4
    assert 0 <= q.correct_index <= 3
    assert q.source == "model"
    assert q.difficulty == "easy"

def test_generate_question_rejected_by_verification():
    reader = MagicMock()
    reader.top_concepts.return_value = ["gravity"]

    reasoner = MagicMock()
    # First call: generation; second call: verification returns NO
    reasoner.reason.side_effect = [
        '{"question": "Q?", "options": ["A","B","C","D"], "correct_index": 0, "explanation": "E"}',
        "NO"
    ]

    agent = QuizAgent(reader, reasoner)
    q = agent.generate_question(topic="gravity", difficulty="easy")
    assert q is None
