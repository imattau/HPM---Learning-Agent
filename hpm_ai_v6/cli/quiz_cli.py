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


def _search_wikipedia_title(query: str) -> Optional[str]:
    """Use Wikipedia query search to resolve a query to the best matching article title."""
    # Try different variations of the query for robustness
    search_terms = [query]
    if " " in query:
        # Also try just the last word (often the subject of the answer)
        search_terms.append(query.split()[-1])
        # And the first two words
        search_terms.append(" ".join(query.split()[:2]))

    for term in search_terms:
        params = urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": term,
            "srlimit": "1",
            "format": "json"
        })
        url = f"https://en.wikipedia.org/w/api.php?{params}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HPM-QuizCLI/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            search_results = data.get("query", {}).get("search", [])
            if search_results:
                return search_results[0]["title"]
        except Exception:
            continue
    return None


def _fetch_wikipedia_full(title: str, max_sentences: int = 40) -> list[str]:
    """Fetch full article extract for *title* and return up to max_sentences sentences."""
    params = urllib.parse.urlencode({
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json"
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HPM-QuizCLI/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return []
        page = next(iter(pages.values()))
        extract = page.get("extract", "")
        sentences = [s.strip() for s in extract.split(".") if len(s.strip()) > 20]
        return sentences[:max_sentences]
    except Exception:
        return []


def _fetch_wikipedia_related_titles(title: str, limit: int = 2) -> list[str]:
    """Fetch up to *limit* related article titles from the article's links."""
    params = urllib.parse.urlencode({
        "action": "query",
        "prop": "links",
        "titles": title,
        "pllimit": "5",
        "plnamespace": "0",
        "format": "json"
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HPM-QuizCLI/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return []
        page = next(iter(pages.values()))
        links = page.get("links", [])
        return [lnk["title"] for lnk in links[:limit]]
    except Exception:
        return []


def _fetch_wikipedia_sentences(topic: str, max_sentences: int = 40) -> list[str]:
    """Search for *topic*, resolve to best article title, fetch full article + 2 related articles."""
    title = _search_wikipedia_title(topic)
    if not title:
        return []

    all_sentences: list[str] = []

    # Full article for the primary title
    primary = _fetch_wikipedia_full(title, max_sentences=max_sentences)
    if not primary:
        print(yellow(f"  [Wikipedia fetch failed for '{topic}' (title: '{title}')]"))
        return []
    all_sentences.extend(primary)

    # Follow up to 2 related article links and fetch their intros
    related_titles = _fetch_wikipedia_related_titles(title, limit=2)
    for rel_title in related_titles:
        rel_sentences = _fetch_wikipedia_full(rel_title, max_sentences=10)
        all_sentences.extend(rel_sentences)

    return all_sentences


def _fetch_wordnet_sentences(word: str, max_senses: int = 4) -> list[str]:
    """Use NLTK WordNet to get definitions and example sentences for a word."""
    try:
        from nltk.corpus import wordnet
        synsets = wordnet.synsets(word.lower().replace(" ", "_"))[:max_senses]
        sentences = []
        for syn in synsets:
            pos = {"n": "noun", "v": "verb", "a": "adjective", "r": "adverb", "s": "adjective"}.get(syn.pos(), "word")
            defn = syn.definition()
            if defn:
                sentences.append(f"{word.title()} ({pos}): {defn}.")
            for example in syn.examples()[:2]:
                sentences.append(example.capitalize() + ".")
        return sentences
    except Exception:
        return []


def train_on_weak_topics(reader, weak_topics: list[tuple[str, str]], dataset_agent=None, min_entropy: float = 0.1) -> None:
    """Fetch Wikipedia text and dictionary definitions for each weak topic, filter by novelty, and retrain.

    Each entry is (display_label, search_query). Sentences are scored by entropy
    (how surprising they are to the model) and only novel ones are trained on.
    """
    if not weak_topics:
        return

    labels = ", ".join(label for label, _ in weak_topics)
    print(f"\nTriggering knowledge acquisition for: {labels}")
    all_sentences: list[str] = []
    for label, query in weak_topics:
        print(f"  Acquiring '{label}' ...", end=" ", flush=True)
        
        # 1. Fetch Encyclopedia Context
        wiki_sentences = _fetch_wikipedia_sentences(query)
        
        # 2. Fetch Lexical Grounding
        dict_sentences = _fetch_wordnet_sentences(label)
        
        sentences = wiki_sentences + dict_sentences
        if not sentences:
            print("(no data)")
            continue

        if dataset_agent is not None:
            # Score each sentence by entropy — skip ones the model already knows
            novel = [s for s in sentences if dataset_agent.score_sentence(s) >= min_entropy]
            skipped = len(sentences) - len(novel)
            print(f"{len(novel)} novel sentences ({skipped} already known, skipped)")
            all_sentences.extend(novel)
        else:
            print(f"{len(sentences)} sentences")
            all_sentences.extend(sentences)

    if all_sentences:
        reader.train_sequence(all_sentences, enable_causal=False)
        print(green(f"Training complete — {len(all_sentences)} sentences processed."))
    else:
        print(yellow("No novel sentences to train on — model already knows this content."))


def _nominate_uncertain_topics(dataset_agent, n: int = 4, pool_size: int = 20,
                               exclude: set[str] | None = None) -> list[str]:
    """Nominate topics the model is most uncertain about (highest prediction entropy).

    Gets a pool of candidate topics from the dataset agent, scores each with a
    probe sentence, ranks by descending score (high entropy = uncertain), and
    returns the top *n* that haven't been nominated before.
    """
    candidates = dataset_agent.generate_wikipedia_topics(max_topics=pool_size)
    if not candidates:
        return []

    if exclude:
        candidates = [t for t in candidates if t not in exclude]

    scored: list[tuple[float, str]] = []
    for topic in candidates:
        try:
            probe = f"{topic} is a subject worth understanding."
            score = dataset_agent.score_sentence(probe)
        except Exception:
            score = 0.0
        scored.append((score, topic))

    # Descending by score: high entropy → model is most uncertain
    scored.sort(key=lambda x: x[0], reverse=True)
    return [topic for _, topic in scored[:n]]


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
    nominated_history: set[str] = set()  # all topics ever nominated across rounds
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
            train_on_weak_topics(reader, weak_topics, dataset_agent=dataset_agent)
            reasoning_agent.invalidate()
        else:
            print(green("All topics answered confidently — no retraining needed."))

        # Let the model nominate its own next learning topics — ranked by uncertainty
        print("\nAsking model what it needs to learn next...")
        model_topics = _nominate_uncertain_topics(
            dataset_agent, n=4, pool_size=20, exclude=nominated_history
        )
        if model_topics:
            nominated_history.update(model_topics)
            print(f"Model nominated (by uncertainty): {', '.join(model_topics)}")
            train_on_weak_topics(reader, [(t, t) for t in model_topics], dataset_agent=dataset_agent)
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


def _score_options(reasoning_agent, question: str, options_map: dict) -> tuple[str, bool, dict]:
    """Score each option by direct edge lookup in the pattern graph.

    For each option, counts edges from its node to question-keyword nodes.
    The option with the most/strongest connections to question terms wins.
    Falls back to reason_with_trace if graph has no edges at all.
    """
    reasoning_agent._ensure_fresh()
    edge_index = getattr(reasoning_agent, "_edge_index", {})

    # Extract meaningful keywords from the question
    q_words = {w for w in question.lower().split()
               if w not in _STOP_WORDS and len(w) > 2 and w.isalpha()}

    scores: dict[str, float] = {}
    for key, option_text in options_map.items():
        if not option_text:
            scores[key] = 0.0
            continue
        option_words = [w.lower() for w in option_text.split() if w.isalpha()]
        total = 0.0
        for opt_word in option_words:
            # Find all edge index keys that contain this option word
            for edge_key, records in edge_index.items():
                if opt_word not in edge_key.lower():
                    continue
                # Score edges whose targets connect to question keywords
                for rec in records:
                    target_key = getattr(rec, "target_key", "") or ""
                    target_name = target_key.split(":")[-1].lower()
                    if any(qw in target_name for qw in q_words):
                        total += float(getattr(rec, "score", getattr(rec, "raw_weight", 0.0)))
        scores[key] = total

    best_key = max(scores, key=lambda k: scores[k])
    best_score = scores[best_key]
    any_confident = best_score > 0.0

    # Get a trace for the winning option for reinforcement/display
    best_trace = reasoning_agent.reason_with_trace(f"{options_map[best_key]} {question}")

    return best_key, any_confident, best_trace


def _parse_letter_answer(answer_text: str, options_map: dict) -> str:
    """Extract the first A/B/C/D letter from the reasoning agent's answer text.
    Falls back to matching option values in the text, then A as last resort."""
    import re as _re
    # Look for a standalone letter A-D at the start of the answer
    match = _re.search(r'\b([A-D])\b', answer_text.upper())
    if match:
        return match.group(1)
    # Try matching option values in the answer text
    answer_lower = answer_text.lower()
    for key, value in options_map.items():
        if value and value.lower() in answer_lower:
            return key
    return "A"


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

        # Score each option independently — pick the one with the most graph support
        chosen, confident, trace = _score_options(reasoning_agent, q.question, options_map)
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
