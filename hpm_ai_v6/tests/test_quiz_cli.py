# hpm_ai_v6/tests/test_quiz_cli.py
import sys
from unittest.mock import patch
import types
from pathlib import Path

def test_parse_args_defaults():
    from hpm_ai_v6.cli.quiz_cli import parse_args
    args = parse_args([])
    assert args.source == "bank"
    assert args.difficulty == "easy"
    assert args.n == 5
    assert args.auto is False

def test_parse_args_custom():
    from hpm_ai_v6.cli.quiz_cli import parse_args
    args = parse_args(["--source", "arc_mmlu", "--difficulty", "hard", "--n", "10", "--auto", "--kiwix-zim-path", "/tmp/offline.zim"])
    assert args.source == "arc_mmlu"
    assert args.difficulty == "hard"
    assert args.n == 10
    assert args.auto is True
    assert args.kiwix_zim_path == "/tmp/offline.zim"

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

    question = _make_mock_question(correct_index=1)
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


def test_run_quiz_reveals_answer_and_emits_feedback(capsys):
    from hpm_ai_v6.cli.quiz_cli import run_quiz

    mock_reader = MagicMock()
    mock_quiz_agent = MagicMock()
    mock_reasoning_agent = MagicMock()

    question = _make_mock_question(correct_index=1)
    mock_quiz_agent.generate_quiz.return_value = [question]
    mock_reader.agents = {}
    mock_reasoning_agent.reason_with_trace.return_value = {
        "candidate_paths": [],
        "chosen_path": None,
        "explanation": "I don't know.",
        "answer": "C",
    }

    feedback = []

    score, weak, mastered = run_quiz(
        reader=mock_reader,
        quiz_agent=mock_quiz_agent,
        reasoning_agent=mock_reasoning_agent,
        n=1,
        difficulty="easy",
        source="bank",
        auto=True,
        feedback_hook=lambda payload: feedback.append(payload),
    )

    assert score == 0
    assert weak[0][0] == "geography"
    assert feedback and feedback[0]["correct"] == "B"
    assert feedback[0]["query"].endswith("London")
    captured = capsys.readouterr()
    assert "Correct answer: B) London" in captured.out

def test_run_quiz_does_not_default_to_a_when_uncertain(capsys):
    from hpm_ai_v6.cli.quiz_cli import run_quiz

    mock_reader = MagicMock()
    mock_quiz_agent = MagicMock()
    mock_reasoning_agent = MagicMock()

    question = _make_mock_question(correct_index=1)
    mock_quiz_agent.generate_quiz.return_value = [question]
    mock_reader.agents = {}
    mock_reasoning_agent.reason_with_trace.return_value = {
        "candidate_paths": [],
        "chosen_path": None,
        "explanation": "",
        "answer": "",
    }

    feedback = []

    score, weak, mastered = run_quiz(
        reader=mock_reader,
        quiz_agent=mock_quiz_agent,
        reasoning_agent=mock_reasoning_agent,
        n=1,
        difficulty="easy",
        source="bank",
        auto=True,
        feedback_hook=lambda payload: feedback.append(payload),
    )

    assert score == 0
    assert weak[0][0] == "geography"
    assert feedback and feedback[0]["chosen"] == "?"
    captured = capsys.readouterr()
    assert "AI answers: ?" in captured.out

def test_run_quiz_prefers_learned_score_over_stale_trace(capsys):
    from hpm_ai_v6.cli.quiz_cli import run_quiz

    mock_reader = MagicMock()
    mock_quiz_agent = MagicMock()
    mock_reasoning_agent = MagicMock()

    question = _make_mock_question(correct_index=0)
    mock_quiz_agent.generate_quiz.return_value = [question]

    mock_agent = MagicMock()
    mock_pager = MagicMock()
    mock_pager.iter_index_payloads.return_value = [
        {"name": "what_is_the_capital_of_france_paris", "weight": 2.0}
    ]
    mock_agent.pattern_pager = mock_pager
    mock_reader.agents = {"test": mock_agent}

    mock_reasoning_agent.reason_with_trace.return_value = {
        "candidate_paths": [],
        "chosen_path": None,
        "explanation": "I think the answer is London.",
        "answer": "C",
        "confidence": 0.9,
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
    captured = capsys.readouterr()
    assert "AI answers: A" in captured.out


def test_run_quiz_learns_exact_answer_memory_after_feedback(capsys):
    from hpm_ai_v6.cli.quiz_cli import run_quiz, _quiz_memory_name

    mock_reader = MagicMock()
    mock_quiz_agent = MagicMock()
    mock_reasoning_agent = MagicMock()

    question = MagicMock()
    question.question = "What distinguishes Type I from Type II errors in hypothesis testing?"
    question.options = [
        "Type I is false negative, Type II is false positive",
        "Type I is false positive, Type II is false negative",
        "Type I tests means, Type II tests variances",
        "Type I uses z-tests, Type II uses t-tests",
    ]
    question.correct_index = 1
    question.topic = "statistics"
    mock_quiz_agent.generate_quiz.return_value = [question]

    mock_agent = MagicMock()
    mock_pager = MagicMock()
    mock_pager.iter_index_payloads.return_value = []
    mock_agent.pattern_pager = mock_pager
    mock_reader.agents = {"test": mock_agent}

    def _feedback(payload):
        if payload.get("correct") == "B":
            mock_pager.iter_index_payloads.return_value = [
                {"name": _quiz_memory_name(question.question, "B"), "weight": 100.0}
            ]

    first_score, first_weak, first_mastered = run_quiz(
        reader=mock_reader,
        quiz_agent=mock_quiz_agent,
        reasoning_agent=mock_reasoning_agent,
        n=1,
        difficulty="easy",
        source="bank",
        auto=True,
        feedback_hook=_feedback,
    )

    second_score, second_weak, second_mastered = run_quiz(
        reader=mock_reader,
        quiz_agent=mock_quiz_agent,
        reasoning_agent=mock_reasoning_agent,
        n=1,
        difficulty="easy",
        source="bank",
        auto=True,
        feedback_hook=_feedback,
    )

    assert first_score == 0
    assert second_score == 1
    captured = capsys.readouterr()
    assert "AI answers: ?" in captured.out
    assert "AI answers: B" in captured.out

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


def test_fetch_wikipedia_sentences_falls_back_to_kiwix(tmp_path):
    from hpm_ai_v6.cli import quiz_cli

    zim_path = tmp_path / "offline.zim"
    zim_path.write_bytes(b"")

    class FakeItem:
        content = b"<html><body>Offline article. It contains useful facts.</body></html>"

    class FakeEntry:
        def get_item(self):
            return FakeItem()

    class FakeArchive:
        def __init__(self, path):
            self.path = path

        def get_entry_by_path(self, path):
            return FakeEntry()

    class FakeQuery:
        def set_query(self, query):
            self.query = query
            return self

    class FakeSearch:
        def getEstimatedMatches(self):
            return 1

        def getResults(self, start, count):
            return ["/offline"]

    class FakeSearcher:
        def __init__(self, archive):
            self.archive = archive

        def search(self, query):
            return FakeSearch()

    libzim = types.ModuleType("libzim")
    reader_mod = types.ModuleType("libzim.reader")
    search_mod = types.ModuleType("libzim.search")
    reader_mod.Archive = FakeArchive
    search_mod.Query = FakeQuery
    search_mod.Searcher = FakeSearcher

    with patch.object(quiz_cli, "_search_wikipedia_title", return_value=None), \
         patch.dict(sys.modules, {"libzim": libzim, "libzim.reader": reader_mod, "libzim.search": search_mod}):
        sentences = quiz_cli._fetch_wikipedia_sentences("offline topic", zim_path=str(zim_path))

    assert any("useful facts" in sentence for sentence in sentences)


def test_fixture_zim_covers_core_topics():
    from hpm_ai_v6.cli.quiz_cli import _fetch_zip_kiwix_sentences

    zim_path = Path(__file__).resolve().parents[1] / "data" / "kiwix" / "wikipedia-mini.zim"
    sentences = _fetch_zip_kiwix_sentences("France capital Paris", str(zim_path))

    assert any("capital is Paris" in sentence or "capital city of France" in sentence for sentence in sentences)
    assert any("France is a country" in sentence for sentence in sentences)


def test_expand_acquisition_terms_builds_variants():
    from hpm_ai_v6.cli.quiz_cli import _expand_acquisition_terms

    variants = _expand_acquisition_terms("geography", "capital france Paris")
    assert "capital france Paris" in variants
    assert "geography" in variants
    assert any(variant == "capital france" for variant in variants)
    assert any(variant.lower() == "france paris" for variant in variants)

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


def test_train_on_weak_topics_partial_timeout_still_trains():
    from concurrent.futures import TimeoutError as FuturesTimeout
    from hpm_ai_v6.cli.quiz_cli import train_on_weak_topics

    mock_reader = MagicMock()

    class FakeFuture:
        def __init__(self, payload):
            self._payload = payload

        def result(self, timeout=0):
            return self._payload

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            self._futures = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            future = FakeFuture(fn(*args, **kwargs))
            self._futures.append(future)
            return future

    def fake_as_completed(futures, timeout=None):
        futures = list(futures)
        if futures:
            yield futures[0]
        raise FuturesTimeout()

    with patch("hpm_ai_v6.cli.quiz_cli._fetch_wikipedia_sentences") as mock_wiki, \
         patch("hpm_ai_v6.cli.quiz_cli._fetch_dictionary_sentences") as mock_dict, \
         patch("concurrent.futures.ThreadPoolExecutor", FakeExecutor), \
         patch("concurrent.futures.as_completed", side_effect=fake_as_completed):
        mock_wiki.side_effect = lambda query, fetched_titles=None: [f"wiki:{query}"]
        mock_dict.side_effect = lambda label: [f"dict:{label}"]
        train_on_weak_topics(
            mock_reader,
            [("geography", "Paris capital"), ("science", "atoms")],
        )

    mock_reader.train_sequence.assert_called_once()
    call_args = mock_reader.train_sequence.call_args[0][0]
    assert "wiki:Paris capital" in call_args
    assert "dict:Paris capital" in call_args
    assert not any("science" in sentence for sentence in call_args)

def test_train_on_weak_topics_empty_list():
    from hpm_ai_v6.cli.quiz_cli import train_on_weak_topics

    mock_reader = MagicMock()
    train_on_weak_topics(mock_reader, [])
    mock_reader.train_sequence.assert_not_called()


def test_main_runs_single_round_and_flushes():
    from argparse import Namespace
    from hpm_ai_v6.cli import quiz_cli

    mock_reader = MagicMock()
    mock_reader.reasoning_agent = MagicMock()
    mock_reader.flush_all = MagicMock()

    mock_quiz_agent = MagicMock()

    with patch.object(quiz_cli, "parse_args", return_value=Namespace(source="bank", difficulty="easy", n=1, auto=True, loop=False)), \
         patch.object(quiz_cli, "_corpus_path", return_value="/tmp/fake_corpus.txt"), \
         patch("hpm_ai_v6.agents.multi_agent_reader.MultiAgentReader", return_value=mock_reader), \
         patch("hpm_ai_v6.agents.quiz_agent.QuizAgent", return_value=mock_quiz_agent), \
         patch.object(quiz_cli, "run_quiz", return_value=(0, [("fallback", "fallback query")], set())):
        quiz_cli.main([])

    mock_quiz_agent.generate_quiz.assert_not_called()
    mock_reader.flush_all.assert_called_once()


def test_main_loops_until_mastered():
    from argparse import Namespace
    from hpm_ai_v6.cli import quiz_cli

    mock_reader = MagicMock()
    mock_reader.reasoning_agent = MagicMock()
    mock_reader.flush_all = MagicMock()

    mock_quiz_agent = MagicMock()

    with patch.object(quiz_cli, "parse_args", return_value=Namespace(source="bank", difficulty="easy", n=1, auto=True, loop=True)), \
         patch.object(quiz_cli, "_corpus_path", return_value="/tmp/fake_corpus.txt"), \
         patch("hpm_ai_v6.agents.multi_agent_reader.MultiAgentReader", return_value=mock_reader), \
         patch("hpm_ai_v6.agents.quiz_agent.QuizAgent", return_value=mock_quiz_agent), \
         patch.object(quiz_cli, "run_quiz", side_effect=[(0, [], set()), (1, [], {"q2"})]):
        quiz_cli.main([])

    assert mock_quiz_agent.generate_quiz.call_count == 0
    assert mock_reader.flush_all.call_count == 2
