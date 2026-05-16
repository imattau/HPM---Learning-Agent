"""CLI quiz tool for the HPM learning agent."""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def green(text: str) -> str:
    return f"{GREEN}{text}{RESET}"

def red(text: str) -> str:
    return f"{RED}{text}{RESET}"

def yellow(text: str) -> str:
    return f"{YELLOW}{text}{RESET}"


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HPM AI general-knowledge quiz (terminal mode)"
    )
    parser.add_argument(
        "--source",
        choices=["bank", "model"],
        default="bank",
        help="Question source: 'bank' (default) or 'model' (AI-generated)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="easy",
        help="Question difficulty (default: easy)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of questions (default: 5)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Skip 'Press Enter' pauses; fully autonomous run",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Corpus path (mirrors web_demo._corpus_path)
# ---------------------------------------------------------------------------
def _corpus_path() -> str:
    # hpm_ai_v6/cli/quiz_cli.py → up two levels → package root
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base, "data", "corpus", "alice_mini.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus not found at {path}")
    return path


# ---------------------------------------------------------------------------
# Stub main (filled out in Tasks 2 & 3)
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    print(f"Quiz CLI — source={args.source} difficulty={args.difficulty} n={args.n} auto={args.auto}")


if __name__ == "__main__":
    main()
