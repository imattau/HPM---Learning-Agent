"""CLI quiz tool for the HPM learning agent."""
from __future__ import annotations

import argparse
import os
import re
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
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep repeating quiz until all answers are correct and confident",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Corpus path (mirrors web_demo._corpus_path)
# ---------------------------------------------------------------------------
def _corpus_path() -> str:
    # hpm_ai_v6/cli/quiz_cli.py → up two levels → hpm_ai_v6
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "corpus", "alice_mini.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus not found at {path}")
    return path


# ---------------------------------------------------------------------------
# Stub main (filled out in Tasks 2 & 3)
# ---------------------------------------------------------------------------
import urllib.request
import urllib.parse
import json


def _search_wikipedia_title(query: str) -> str:
    """Use Wikipedia opensearch to resolve a query to the best matching article title.
    Tries the last word (correct answer) first, then the full query as fallback."""
    candidates = [query.split()[-1], query] if " " in query else [query]
    for candidate in candidates:
        params = urllib.parse.urlencode({
            "action": "opensearch", "search": candidate, "limit": "1", "format": "json"
        })
        url = f"https://en.wikipedia.org/w/api.php?{params}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HPM-QuizCLI/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            titles = data[1] if len(data) > 1 else []
            if titles:
                return titles[0]
        except Exception:
            pass
    return query


def _fetch_wikipedia_sentences(topic: str, max_sentences: int = 20) -> list[str]:
    """Search for *topic*, resolve to best article title, fetch intro sentences."""
    title = _search_wikipedia_title(topic)
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HPM-QuizCLI/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        extract = data.get("extract", "")
        sentences = [s.strip() for s in extract.split(".") if len(s.strip()) > 20]
        return sentences[:max_sentences]
    except Exception as exc:
        print(yellow(f"  [Wikipedia fetch failed for '{topic}': {exc}]"))
        return []


def train_on_weak_topics(reader, weak_topics: list[tuple[str, str]]) -> None:
    """Fetch Wikipedia text for each weak topic and retrain the reader.

    Each entry is (display_label, search_query) where search_query is the
    correct answer text — much more specific than the coarse topic label.
    """
    if not weak_topics:
        return

    labels = ", ".join(label for label, _ in weak_topics)
    print(f"\nTriggering Wikipedia training on: {labels}")
    all_sentences: list[str] = []
    for label, query in weak_topics:
        print(f"  Fetching Wikipedia: '{query}' ...", end=" ", flush=True)
        sentences = _fetch_wikipedia_sentences(query)
        if sentences:
            print(f"{len(sentences)} sentences")
            all_sentences.extend(sentences)
        else:
            print("(no data)")

    if all_sentences:
        reader.train_sequence(all_sentences, enable_causal=False)
        print(green(f"Training complete — {len(all_sentences)} sentences processed."))
    else:
        print(yellow("No Wikipedia data retrieved; skipping training."))


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    print("Building HPM reader...")
    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
    from hpm_ai_v6.agents.quiz_agent import QuizAgent
    from hpm_ai_v6.agents.dataset_training_agent import DatasetTrainingAgent

    corpus = _corpus_path()
    reader = MultiAgentReader(corpus, warm_start=True, warm_start_limit=500)
    reasoning_agent = getattr(reader, "reasoning_agent", None)
    if reasoning_agent is None:
        print(red("Error: reasoning_agent not found on MultiAgentReader."))
        sys.exit(1)

    quiz_agent = QuizAgent(reader, reasoning_agent)
    dataset_agent = DatasetTrainingAgent(reader, corpus_path=corpus)

    mastered: set[str] = set()  # question ids answered correctly + confidently
    round_num = 0

    while True:
        round_num += 1
        if args.loop and round_num > 1:
            print(f"\n{yellow(f'=== Loop round {round_num} ===')} (mastered {len(mastered)} question(s) so far)")

        score, weak_topics, newly_mastered = run_quiz(
            reader=reader,
            quiz_agent=quiz_agent,
            reasoning_agent=reasoning_agent,
            n=args.n,
            difficulty=args.difficulty,
            source=args.source,
            auto=args.auto,
            skip_ids=mastered,
        )
        mastered.update(newly_mastered)

        if weak_topics:
            train_on_weak_topics(reader, weak_topics)
            reasoning_agent.invalidate()
        else:
            print(green("All topics answered confidently — no retraining needed."))

        # Let the model nominate its own next learning topics from pattern gaps
        print("\nAsking model what it needs to learn next...")
        model_topics = dataset_agent.generate_wikipedia_topics(max_topics=4)
        if model_topics:
            print(f"Model nominated: {', '.join(model_topics)}")
            train_on_weak_topics(reader, [(t, t) for t in model_topics])
            reasoning_agent.invalidate()
        else:
            print(yellow("Model could not nominate topics yet."))

        if not args.loop:
            break

        remaining = args.n - len(mastered)
        if remaining <= 0:
            print(green(f"\nAll {args.n} questions mastered! Quiz complete."))
            break
        print(f"\n{remaining} question(s) still to master. Rerunning...")


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


_STOP_WORDS = {"what", "is", "the", "a", "an", "of", "in", "on", "at", "to", "for",
               "are", "was", "were", "does", "do", "how", "many", "which", "who",
               "where", "when", "why", "that", "this", "it", "its", "have", "has"}

def _build_search_query(question: str, correct_answer: str) -> str:
    """Build a specific Wikipedia search query from question keywords + correct answer."""
    # Extract meaningful words from the question (drop stop words and punctuation)
    words = re.sub(r"[^\w\s]", " ", question.lower()).split()
    keywords = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    # Combine up to 4 question keywords with the correct answer
    query_parts = keywords[:4] + [correct_answer]
    return " ".join(query_parts)


def _reinforce_trace(reasoning_agent, trace: dict, boost: bool) -> None:
    """Strengthen (boost=True) or weaken (boost=False) edges from a reasoning trace."""
    chosen_path = trace.get("chosen_path")
    if not chosen_path:
        return
    steps = chosen_path if isinstance(chosen_path, list) else [chosen_path]
    for step in steps:
        source = getattr(step, "source", None)
        target = getattr(step, "target", None)
        score = getattr(step, "score", 0.0)
        if source is None or target is None:
            continue
        new_score = float(score) * 1.5 if boost else float(score) * 0.3
        try:
            reasoning_agent.promote_reasoning_edge(source, target, max(new_score, 1e-6))
        except Exception:
            pass
    try:
        reasoning_agent.invalidate()
    except Exception:
        pass


def run_quiz(
    reader,
    quiz_agent,
    reasoning_agent,
    n: int,
    difficulty: str,
    source: str,
    auto: bool,
    skip_ids: set = None,
) -> tuple[int, list[str], set]:
    """Run the quiz loop. Returns (score, weak_topics, newly_mastered_ids)."""
    questions = quiz_agent.generate_quiz(n=n, difficulty=difficulty, source=source)
    if skip_ids:
        questions = [q for q in questions if q.id not in skip_ids]
    score = 0
    weak_topics: list[tuple[str, str]] = []  # (display_label, search_query)
    weak_seen: set[str] = set()
    newly_mastered: set = set()

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
            newly_mastered.add(q.id)
            # Reinforce the edges that led to this correct answer
            _reinforce_trace(reasoning_agent, trace, boost=True)
        elif is_correct and not confident:
            # Lucky guess — still correct but treat topic as weak
            print(yellow("Correct (lucky guess)"))
            score += 1
            topic = getattr(q, "topic", None)
            if topic and topic not in weak_seen:
                weak_seen.add(topic)
                weak_topics.append((topic, _build_search_query(q.question, options_map.get(correct, ""))))
        else:
            print(red(f"Incorrect. Correct answer: {correct}) {options_map.get(correct, '')}"))
            topic = getattr(q, "topic", None)
            if topic and topic not in weak_seen:
                weak_seen.add(topic)
                weak_topics.append((topic, _build_search_query(q.question, options_map.get(correct, ""))))
            # Weaken the edges that led to this wrong confident answer
            if confident:
                _reinforce_trace(reasoning_agent, trace, boost=False)

        if not auto and idx < len(questions):
            input("\nPress Enter for next question...")

    print(f"\n{'='*60}")
    print(f"Final score: {score}/{len(questions)}")
    if weak_topics:
        print(yellow(f"Weak topics: {', '.join(label for label, _ in weak_topics)}"))
    else:
        print(green("No weak topics identified."))

    return score, weak_topics, newly_mastered


if __name__ == "__main__":
    main()
