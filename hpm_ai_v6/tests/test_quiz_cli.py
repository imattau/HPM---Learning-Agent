# hpm_ai_v6/tests/test_quiz_cli.py
import sys
from unittest.mock import patch

def test_parse_args_defaults():
    from hpm_ai_v6.cli.quiz_cli import parse_args
    args = parse_args([])
    assert args.source == "bank"
    assert args.difficulty == "easy"
    assert args.n == 5
    assert args.auto is False

def test_parse_args_custom():
    from hpm_ai_v6.cli.quiz_cli import parse_args
    args = parse_args(["--source", "model", "--difficulty", "hard", "--n", "10", "--auto"])
    assert args.source == "model"
    assert args.difficulty == "hard"
    assert args.n == 10
    assert args.auto is True

from unittest.mock import MagicMock, patch
import io

def _make_mock_question(correct_index=0):
    q = MagicMock()
    q.question = "What is the capital of France?"
    q.options = ["Paris", "London", "Berlin", "Madrid"]
    q.correct_index = correct_index
    q.topic = "geography"
    return q

def test_run_quiz_correct_answer(capsys):
    from hpm_ai_v6.cli.quiz_cli import run_quiz

    mock_reader = MagicMock()
    mock_quiz_agent = MagicMock()
    mock_reasoning_agent = MagicMock()

    question = _make_mock_question(correct_index=0)
    mock_quiz_agent.generate_quiz.return_value = [question]

    # reasoning_agent returns trace pointing to option A
    mock_reasoning_agent.reason_with_trace.return_value = {
        "candidate_paths": [{"label": "Paris"}],
        "chosen_path": {"label": "Paris"},
        "explanation": "Paris is the capital of France.",
        "answer": "A",
    }

    score, weak = run_quiz(
        reader=mock_reader,
        quiz_agent=mock_quiz_agent,
        reasoning_agent=mock_reasoning_agent,
        n=1,
        difficulty="easy",
        source="bank",
        auto=True,
    )

    assert score == 1
    assert weak == []
    captured = capsys.readouterr()
    assert "Paris" in captured.out or "correct" in captured.out.lower()

def test_run_quiz_wrong_answer_adds_weak_topic(capsys):
    from hpm_ai_v6.cli.quiz_cli import run_quiz

    mock_reader = MagicMock()
    mock_quiz_agent = MagicMock()
    mock_reasoning_agent = MagicMock()

    question = _make_mock_question(correct_index=0)
    mock_quiz_agent.generate_quiz.return_value = [question]

    mock_reasoning_agent.reason_with_trace.return_value = {
        "candidate_paths": [],
        "chosen_path": None,
        "explanation": "I don't know.",
        "answer": "C",
    }

    score, weak = run_quiz(
        reader=mock_reader,
        quiz_agent=mock_quiz_agent,
        reasoning_agent=mock_reasoning_agent,
        n=1,
        difficulty="easy",
        source="bank",
        auto=True,
    )

    assert score == 0
    assert "geography" in weak

def test_train_on_weak_topics_calls_train_sequence():
    from hpm_ai_v6.cli.quiz_cli import train_on_weak_topics

    mock_reader = MagicMock()

    # Patch urllib fetch to return fake Wikipedia text
    fake_wiki = "France is a country in Western Europe. Paris is its capital."
    with patch("hpm_ai_v6.cli.quiz_cli._fetch_wikipedia_sentences") as mock_fetch:
        mock_fetch.return_value = fake_wiki.split(". ")
        train_on_weak_topics(mock_reader, ["geography"])

    mock_reader.train_sequence.assert_called_once()
    call_args = mock_reader.train_sequence.call_args[0][0]
    assert isinstance(call_args, list)
    assert len(call_args) > 0

def test_train_on_weak_topics_empty_list():
    from hpm_ai_v6.cli.quiz_cli import train_on_weak_topics

    mock_reader = MagicMock()
    train_on_weak_topics(mock_reader, [])
    mock_reader.train_sequence.assert_not_called()
