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


def _wiki_get(url: str, timeout: int = 6) -> Optional[dict]:
    """Make a Wikipedia API GET with a hard socket timeout. Returns parsed JSON or None."""
    import socket
    old_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(timeout)
        req = urllib.request.Request(url, headers={"User-Agent": "HPM-QuizCLI/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None
    finally:
        socket.setdefaulttimeout(old_timeout)


def _search_wikipedia_title(query: str) -> Optional[str]:
    """Use Wikipedia query search to resolve a query to the best matching article title."""
    search_terms = [query]
    if " " in query:
        search_terms.append(query.split()[-1])
        search_terms.append(" ".join(query.split()[:2]))

    for term in search_terms:
        params = urllib.parse.urlencode({
            "action": "query", "list": "search",
            "srsearch": term, "srlimit": "1", "format": "json"
        })
        data = _wiki_get(f"https://en.wikipedia.org/w/api.php?{params}")
        if data:
            results = data.get("query", {}).get("search", [])
            if results:
                return results[0]["title"]
    return None


def _fetch_wikipedia_full(title: str, max_sentences: int = 40) -> list[str]:
    """Fetch full article extract for *title* and return up to max_sentences sentences."""
    params = urllib.parse.urlencode({
        "action": "query", "prop": "extracts",
        "explaintext": True, "titles": title, "format": "json"
    })
    data = _wiki_get(f"https://en.wikipedia.org/w/api.php?{params}")
    if not data:
        return []
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return []
    extract = next(iter(pages.values())).get("extract", "")
    return [s.strip() for s in extract.split(".") if len(s.strip()) > 20][:max_sentences]


def _fetch_wikipedia_related_titles(title: str, limit: int = 2) -> list[str]:
    """Fetch up to *limit* related article titles from the article's links."""
    params = urllib.parse.urlencode({
        "action": "query", "prop": "links", "titles": title,
        "pllimit": "5", "plnamespace": "0", "format": "json"
    })
    data = _wiki_get(f"https://en.wikipedia.org/w/api.php?{params}")
    if not data:
        return []
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return []
    links = next(iter(pages.values())).get("links", [])
    return [lnk["title"] for lnk in links[:limit]]


def _fetch_wikipedia_sentences(topic: str, max_sentences: int = 40,
                               fetched_titles: set[str] | None = None) -> list[str]:
    """Fetch Wikipedia content for *topic*, skipping already-fetched titles.

    All article fetches run in parallel threads with a 12s total budget.
    When the primary article is already known, follows links to find new content.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

    if fetched_titles is None:
        fetched_titles = set()

    title = _search_wikipedia_title(topic)
    if not title:
        return []

    # Determine which titles to fetch
    if title not in fetched_titles:
        primary_titles = [title]
    else:
        related = _fetch_wikipedia_related_titles(title, limit=8)
        primary_titles = [t for t in related if t not in fetched_titles][:3]

    if not primary_titles:
        return []

    all_sentences: list[str] = []

    # Fetch all primary titles in parallel
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fetch_wikipedia_full, t, max_sentences): t for t in primary_titles}
        try:
            for fut in as_completed(futures, timeout=12):
                t = futures[fut]
                try:
                    sentences = fut.result(timeout=0)
                    if sentences:
                        fetched_titles.add(t)
                        all_sentences.extend(sentences)
                except Exception:
                    pass
        except FuturesTimeout:
            pass

    # Fire-and-forget: queue second-level link fetches in background (no wait)
    if all_sentences:
        link_titles: list[str] = []
        for t in list(fetched_titles)[:2]:
            for rel in _fetch_wikipedia_related_titles(t, limit=2):
                if rel not in fetched_titles and rel not in link_titles:
                    link_titles.append(rel)

        def _background_fetch(rel_title: str) -> None:
            sents = _fetch_wikipedia_full(rel_title, max_sentences=10)
            if sents:
                fetched_titles.add(rel_title)
                all_sentences.extend(sents)

        with ThreadPoolExecutor(max_workers=2) as pool:
            link_futures = [pool.submit(_background_fetch, t) for t in link_titles[:4]]
            try:
                for fut in as_completed(link_futures, timeout=8):
                    fut.result(timeout=0)
            except FuturesTimeout:
                pass

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


def train_on_weak_topics(reader, weak_topics: list[tuple[str, str]], dataset_agent=None,
                         min_entropy: float = 0.1, fetched_titles: set[str] | None = None) -> None:
    """Fetch Wikipedia text and dictionary definitions for each weak topic, filter by novelty, and retrain.

    Each entry is (display_label, search_query). Sentences are scored by entropy
    (how surprising they are to the model) and only novel ones are trained on.
    """
    if not weak_topics:
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed

    labels = ", ".join(label for label, _ in weak_topics)
    print(f"\nTriggering knowledge acquisition for: {labels}")
    all_sentences: list[str] = []

    def _acquire_topic(label: str, query: str) -> tuple[str, list[str]]:
        wiki = _fetch_wikipedia_sentences(query, fetched_titles=fetched_titles)
        wnet = _fetch_wordnet_sentences(label)
        return label, wiki + wnet

    with ThreadPoolExecutor(max_workers=min(len(weak_topics), 4)) as pool:
        futures = {pool.submit(_acquire_topic, label, query): (label, query)
                   for label, query in weak_topics}
        topic_results: dict[str, list[str]] = {}
        for fut in as_completed(futures, timeout=60):
            try:
                label, sentences = fut.result(timeout=0)
                topic_results[label] = sentences
            except Exception:
                label, _ = futures[fut]
                topic_results[label] = []

    for label, query in weak_topics:
        sentences = topic_results.get(label, [])
        print(f"  Acquiring '{label}' ...", end=" ", flush=True)
        if not sentences:
            print("(no data)")
            continue

        if dataset_agent is not None:
            scored = [(dataset_agent.score_sentence(s), s) for s in sentences]
            novel = [s for sc, s in scored if sc >= min_entropy]
            if novel:
                skipped = len(sentences) - len(novel)
                print(f"{len(novel)} novel sentences ({skipped} already known, skipped)")
                all_sentences.extend(novel)
            else:
                # No novel sentences — try exploring links for fresh content
                link_sentences = _fetch_wikipedia_sentences(query, fetched_titles=fetched_titles)
                link_scored = [(dataset_agent.score_sentence(s), s) for s in link_sentences] if link_sentences else []
                link_novel = [s for sc, s in link_scored if sc >= min_entropy]
                if link_novel:
                    print(f"{len(link_novel)} novel sentences from linked articles")
                    all_sentences.extend(link_novel)
                else:
                    # Still nothing novel — reinforce with highest-entropy subset
                    scored.sort(key=lambda x: x[0], reverse=True)
                    reinforcement = [s for _, s in scored[:5]]
                    print(f"0 novel — reinforcing with {len(reinforcement)} highest-entropy sentences")
                    all_sentences.extend(reinforcement)
        else:
            print(f"{len(sentences)} sentences")
            all_sentences.extend(sentences)

    if all_sentences:
        reader.train_sequence(all_sentences, enable_causal=False)
        print(green(f"Training complete — {len(all_sentences)} sentences processed."))
    else:
        print(yellow("No novel sentences to train on — model already knows this content."))


def _nominate_uncertain_topics(dataset_agent, reasoning_agent, n: int = 4,
                               pool_size: int = 20,
                               exclude: set[str] | None = None) -> list[str]:
    """Nominate topics where the model has the fewest learned pattern graph edges.

    Gets a pool of candidate topics, counts edges in the reasoning graph whose
    keys contain each topic word, ranks by ascending edge count (sparse = least
    learned = highest priority), and returns the top *n* not yet nominated.
    """
    candidates = dataset_agent.generate_wikipedia_topics(max_topics=pool_size)
    if not candidates:
        return []

    if exclude:
        candidates = [t for t in candidates if t not in exclude]

    reasoning_agent._ensure_fresh()
    edge_index = getattr(reasoning_agent, "_edge_index", {})

    scored: list[tuple[int, str]] = []
    for topic in candidates:
        topic_word = topic.lower().split()[0]  # use first word for lookup
        edge_count = sum(1 for k in edge_index if topic_word in k.lower())
        scored.append((edge_count, topic))

    # Ascending by edge count: fewest edges = least learned = learn this first
    scored.sort(key=lambda x: x[0])
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
    fetched_titles: set[str] = set()  # all Wikipedia article titles fetched this session
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
            train_on_weak_topics(reader, weak_topics, dataset_agent=dataset_agent, fetched_titles=fetched_titles)
            reasoning_agent.invalidate()
        else:
            print(green("All topics answered confidently — no retraining needed."))

        # Let the model nominate its own next learning topics — ranked by uncertainty
        print("\nAsking model what it needs to learn next...")
        model_topics = _nominate_uncertain_topics(
            dataset_agent, reasoning_agent, n=4, pool_size=20, exclude=nominated_history
        )
        if model_topics:
            nominated_history.update(model_topics)
            print(f"Model nominated (by uncertainty): {', '.join(model_topics)}")
            train_on_weak_topics(reader, [(t, t) for t in model_topics], dataset_agent=dataset_agent, fetched_titles=fetched_titles)
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
