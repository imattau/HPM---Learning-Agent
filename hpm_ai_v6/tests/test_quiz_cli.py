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

    # Mock an agent with a relevant pattern so _score_options is confident
    mock_agent = MagicMock()
    mock_pager = MagicMock()
    mock_pager.iter_index_payloads.return_value = [
        {"name": "paris_capital_france", "weight": 1.0}
    ]
    mock_agent.pattern_pager = mock_pager
    mock_reader.agents = {"test": mock_agent}

    # reasoning_agent returns trace pointing to option A
    mock_reasoning_agent.reason_with_trace.return_value = {
        "candidate_paths": [{"label": "Paris"}],
        "chosen_path": {"label": "Paris"},
        "explanation": "Paris is the capital of France.",
        "answer": "A",
        "confidence": 1.0,
    }

    score, weak, mastered = run_quiz(
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
    assert len(mastered) == 1
    captured = capsys.readouterr()
    assert "Paris" in captured.out or "correct" in captured.out.lower()

def test_run_quiz_wrong_answer_adds_weak_topic(capsys):
    from hpm_ai_v6.cli.quiz_cli import run_quiz

    mock_reader = MagicMock()
    mock_quiz_agent = MagicMock()
    mock_reasoning_agent = MagicMock()

    question = _make_mock_question(correct_index=0)
    mock_quiz_agent.generate_quiz.return_value = [question]

    # Mock empty agents so it's not confident
    mock_reader.agents = {}

    mock_reasoning_agent.reason_with_trace.return_value = {
        "candidate_paths": [],
        "chosen_path": None,
        "explanation": "I don't know.",
        "answer": "C",
    }

    score, weak, mastered = run_quiz(
        reader=mock_reader,
        quiz_agent=mock_quiz_agent,
        reasoning_agent=mock_reasoning_agent,
        n=1,
        difficulty="easy",
        source="bank",
        auto=True,
    )

    assert score == 0
    assert weak[0][0] == "geography"

import json

def test_fetch_dictionary_sentences():
    from hpm_ai_v6.cli.quiz_cli import _fetch_dictionary_sentences
    
    mock_data = [
        {
            "meanings": [
                {
                    "partOfSpeech": "noun",
                    "definitions": [
                        {"definition": "A large city.", "example": "Paris is a city."}
                    ]
                }
            ]
        }
    ]
    
    with patch("urllib.request.urlopen") as mock_url:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_data).encode()
        mock_resp.__enter__.return_value = mock_resp
        mock_url.return_value = mock_resp
        
        sentences = _fetch_dictionary_sentences("city")
        assert len(sentences) == 2
        assert "defined as: A large city" in sentences[0]
        assert "example of using 'city' is: Paris is a city" in sentences[1]

def test_fetch_wikipedia_sentences():
    from hpm_ai_v6.cli.quiz_cli import _fetch_wikipedia_sentences
    
    mock_search_data = {
        "query": {
            "search": [{"title": "France"}]
        }
    }
    mock_extract_data = {
        "query": {
            "pages": {
                "123": {"extract": "France is a country in Europe. It has a long history."}
            }
        }
    }
    
    with patch("urllib.request.urlopen") as mock_url:
        mock_resp_search = MagicMock()
        mock_resp_search.read.return_value = json.dumps(mock_search_data).encode()
        mock_resp_search.__enter__.return_value = mock_resp_search
        
        mock_resp_extract = MagicMock()
        mock_resp_extract.read.return_value = json.dumps(mock_extract_data).encode()
        mock_resp_extract.__enter__.return_value = mock_resp_extract
        
        # side_effect to return search result then extract result
        mock_url.side_effect = [mock_resp_search, mock_resp_extract]
        
        sentences = _fetch_wikipedia_sentences("France")
        assert len(sentences) == 2
        assert "France is a country" in sentences[0]
        assert "long history" in sentences[1]

def test_train_on_weak_topics_calls_train_sequence():
    from hpm_ai_v6.cli.quiz_cli import train_on_weak_topics

    mock_reader = MagicMock()

    with patch("hpm_ai_v6.cli.quiz_cli._fetch_wikipedia_sentences") as mock_wiki, \
         patch("hpm_ai_v6.cli.quiz_cli._fetch_dictionary_sentences") as mock_dict:
        
        mock_wiki.return_value = ["Wiki sentence."]
        mock_dict.return_value = ["Dict sentence."]
        
        train_on_weak_topics(mock_reader, [("geography", "Paris capital")])

    mock_reader.train_sequence.assert_called_once()
    call_args = mock_reader.train_sequence.call_args[0][0]
    assert "Wiki sentence." in call_args
    assert "Dict sentence." in call_args

def test_train_on_weak_topics_empty_list():
    from hpm_ai_v6.cli.quiz_cli import train_on_weak_topics

    mock_reader = MagicMock()
    train_on_weak_topics(mock_reader, [])
    mock_reader.train_sequence.assert_not_called()
