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
