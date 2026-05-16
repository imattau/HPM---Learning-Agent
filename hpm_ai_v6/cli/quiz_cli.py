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


OPTION_KEYS = ["A", "B", "C", "D"]


def _extract_answer(trace: dict, options: dict) -> str:
    """Map reasoning trace output to an option key A/B/C/D."""
    # If the agent set an explicit 'answer' key, trust it.
    if trace.get("answer") in OPTION_KEYS:
        return trace["answer"]

    # Otherwise match explanation / chosen_path label against option values.
    chosen = trace.get("chosen_path") or {}
    label = str(chosen.get("label", "")).lower()
    explanation = str(trace.get("explanation", "")).lower()

    for key, value in options.items():
        v = value.lower()
        if v and (v in label or v in explanation):
            return key

    # Fallback: first candidate path label
    candidates = trace.get("candidate_paths") or []
    if candidates:
        candidate_label = str(candidates[0].get("label", "")).lower()
        for key, value in options.items():
            if value.lower() in candidate_label:
                return key

    # Last resort: pick A
    return "A"


def run_quiz(
    reader,
    quiz_agent,
    reasoning_agent,
    n: int,
    difficulty: str,
    source: str,
    auto: bool,
) -> tuple[int, list[str]]:
    """Run the quiz loop. Returns (score, weak_topics)."""
    questions = quiz_agent.generate_quiz(n=n, difficulty=difficulty, source=source)
    score = 0
    weak_topics: list[str] = []

    for idx, q in enumerate(questions, start=1):
        print(f"\n{'='*60}")
        print(f"Question {idx}/{len(questions)}: {q.question}")
        
        # Map list options to A/B/C/D dict for display and prompt
        options_map = {OPTION_KEYS[i]: q.options[i] for i in range(len(q.options))}
        for key in OPTION_KEYS:
            option_text = options_map.get(key, "")
            if option_text:
                print(f"  {key}) {option_text}")

        # Build a focused prompt for the reasoning agent
        prompt = (
            f"{q.question} "
            + " ".join(f"{k}: {v}" for k, v in options_map.items() if v)
        )
        trace = reasoning_agent.reason_with_trace(prompt)

        confident = bool(trace.get("candidate_paths") or trace.get("chosen_path"))
        chosen = _extract_answer(trace, options_map)
        correct = OPTION_KEYS[q.correct_index]

        confidence_label = "confident" if confident else yellow("guessing")
        print(f"\nAI answers: {chosen}  [{confidence_label}]")

        snippet = str(trace.get("explanation", ""))[:120]
        if snippet:
            print(f"Reasoning: {snippet}")

        is_correct = chosen == correct
        if is_correct and confident:
            print(green("Correct!"))
            score += 1
        elif is_correct and not confident:
            # Lucky guess — still correct but treat topic as weak
            print(yellow("Correct (lucky guess)"))
            score += 1
            topic = getattr(q, "topic", None)
            if topic and topic not in weak_topics:
                weak_topics.append(topic)
        else:
            print(red(f"Incorrect. Correct answer: {correct}) {options_map.get(correct, '')}"))
            topic = getattr(q, "topic", None)
            if topic and topic not in weak_topics:
                weak_topics.append(topic)

        if not auto and idx < len(questions):
            input("\nPress Enter for next question...")

    print(f"\n{'='*60}")
    print(f"Final score: {score}/{len(questions)}")
    if weak_topics:
        print(yellow(f"Weak topics: {', '.join(weak_topics)}"))
    else:
        print(green("No weak topics identified."))

    return score, weak_topics


if __name__ == "__main__":
    main()
