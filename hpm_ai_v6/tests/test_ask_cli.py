from __future__ import annotations

import json
from argparse import Namespace
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


def test_parse_args_defaults():
    from hpm_ai_v6.cli.ask_cli import parse_args

    args = parse_args([])
    assert args.question is None
    assert args.interactive is False
    assert args.method == "auto"
    assert args.verbose is False
    assert args.json is False


def test_parse_args_custom():
    from hpm_ai_v6.cli.ask_cli import parse_args

    args = parse_args(["--method", "backward", "--verbose", "What connects alice and rabbit?"])
    assert args.question == "What connects alice and rabbit?"
    assert args.method == "backward"
    assert args.verbose is True


def test_run_question_prints_compact_summary():
    from hpm_ai_v6.cli.ask_cli import run_question

    reasoning_agent = MagicMock()
    reasoning_agent.reason_with_trace.return_value = {
        "question": "How does alice connect to rabbit?",
        "intent": "path",
        "mode": "connection",
        "method": "auto",
        "answer": "Alice connects to rabbit through a sentence bridge.",
        "candidate_paths": [
            {
                "nodes": [{"label": "alice"}, {"label": "rabbit"}],
                "combined_score": 0.71,
            }
        ],
        "chosen_path": {
            "nodes": [{"label": "alice"}, {"label": "sentence"}, {"label": "rabbit"}],
            "combined_score": 0.82,
        },
        "evidence": [
            {"relation": "lexical_transition", "agent": "word", "pattern": "w_alice->rabbit", "score": 0.71}
        ],
    }

    stream = StringIO()
    run_question(reasoning_agent, "How does alice connect to rabbit?", stream=stream)

    output = stream.getvalue()
    assert "Answer: Alice connects to rabbit through a sentence bridge." in output
    assert "Intent: path" in output
    assert "Mode: connection" in output
    assert "Chosen path: yes" in output
    assert "Candidate paths:" in output
    assert "alice -> rabbit" in output
    assert "Evidence:" in output
    reasoning_agent.reason_with_trace.assert_called_once_with(
        "How does alice connect to rabbit?",
        method="auto",
    )


def test_run_question_prints_useful_no_path_summary():
    from hpm_ai_v6.cli.ask_cli import run_question

    reasoning_agent = MagicMock()
    reasoning_agent.reason_with_trace.return_value = {
        "question": "Why is this unknown?",
        "intent": "path",
        "mode": "connection",
        "method": "auto",
        "answer": "I could not match enough learned concepts in that question.",
        "candidate_paths": [],
        "chosen_path": None,
        "evidence": [],
    }

    stream = StringIO()
    run_question(reasoning_agent, "Why is this unknown?", stream=stream)

    output = stream.getvalue()
    assert "Chosen path: no" in output
    assert "I could not match enough learned concepts" in output


@pytest.mark.parametrize("terminator", ["quit", "EOF"])
def test_run_interactive_exits_cleanly(terminator):
    from hpm_ai_v6.cli.ask_cli import run_interactive

    reasoning_agent = MagicMock()
    reasoning_agent.reason_with_trace.return_value = {
        "answer": "stub",
        "intent": "path",
        "mode": "connection",
        "method": "auto",
        "candidate_paths": [],
        "chosen_path": None,
        "evidence": [],
    }

    prompts = []

    if terminator == "quit":
        inputs = iter(["quit"])

        def fake_input(prompt):
            prompts.append(prompt)
            return next(inputs)
    else:
        def fake_input(prompt):
            prompts.append(prompt)
            raise EOFError

    stream = StringIO()
    run_interactive(reasoning_agent, input_fn=fake_input, stream=stream)

    assert prompts
    assert reasoning_agent.reason_with_trace.call_count == 0


def test_main_supports_json_output(capsys):
    from hpm_ai_v6.cli import ask_cli

    mock_reader = MagicMock()
    mock_reader.reasoning_agent = MagicMock()
    mock_reader.reasoning_agent.reason_with_trace.return_value = {
        "question": "What is the answer?",
        "intent": "analogy",
        "mode": "default",
        "method": "beam",
        "answer": "stub answer",
        "candidate_paths": [],
        "chosen_path": None,
        "evidence": [],
    }

    with patch.object(
        ask_cli,
        "parse_args",
        return_value=Namespace(
            question="What is the answer?",
            interactive=False,
            method="beam",
            verbose=False,
            json=True,
        ),
    ), patch.object(ask_cli, "build_reader", return_value=mock_reader):
        ask_cli.main([])

    payload = json.loads(capsys.readouterr().out)
    assert payload["answer"] == "stub answer"
    assert mock_reader.reasoning_agent.reason_with_trace.call_args.kwargs["method"] == "beam"
