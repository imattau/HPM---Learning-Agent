"""CLI quiz tool for the HPM learning agent."""
from __future__ import annotations

import argparse
import html as _html
import os
import re
import sys
import zipfile
from typing import Callable, List, Optional

import numpy as np
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.quiz_feedback import learn_from_quiz_attempt


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
        choices=["bank", "arc_mmlu", "model"],
        default="bank",
        help="Question source: 'bank' (default), 'arc_mmlu', or 'model' (AI-generated)",
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
    parser.add_argument(
        "--kiwix-zim-path",
        default=None,
        help="Path to a local Kiwix .zim file or directory containing one for offline acquisition",
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

_KIWIX_ZIM_ENV_VARS = ("HPM_KIWIX_ZIM_PATH", "KIWIX_ZIM_PATH")
_KIWIX_ZIM_OVERRIDE: Optional[str] = None


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


def _kiwix_archive_path(cli_path: Optional[str] = None) -> Optional[str]:
    """Return a local ZIM archive path from env vars, if configured."""
    if cli_path:
        raw_path = cli_path.strip()
        if raw_path:
            if os.path.isdir(raw_path):
                zim_files = sorted(
                    name for name in os.listdir(raw_path) if name.lower().endswith(".zim")
                )
                if zim_files:
                    return os.path.join(raw_path, zim_files[0])
            elif os.path.exists(raw_path):
                return raw_path

    if _KIWIX_ZIM_OVERRIDE:
        raw_path = _KIWIX_ZIM_OVERRIDE.strip()
        if raw_path:
            if os.path.isdir(raw_path):
                zim_files = sorted(
                    name for name in os.listdir(raw_path) if name.lower().endswith(".zim")
                )
                if zim_files:
                    return os.path.join(raw_path, zim_files[0])
            elif os.path.exists(raw_path):
                return raw_path

    for env_name in _KIWIX_ZIM_ENV_VARS:
        raw_path = os.environ.get(env_name, "").strip()
        if not raw_path:
            continue
        if os.path.isdir(raw_path):
            zim_files = sorted(
                name for name in os.listdir(raw_path) if name.lower().endswith(".zim")
            )
            if zim_files:
                return os.path.join(raw_path, zim_files[0])
            continue
        if os.path.exists(raw_path):
            return raw_path
    return None


def _strip_html(text: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = _html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _topic_tokens(topic: str) -> list[str]:
    tokens = [tok.lower() for tok in re.findall(r"[a-zA-Z0-9]+", topic)]
    if " " in topic:
        tokens.append(topic.lower().strip())
    seen = set()
    ordered: list[str] = []
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def _expand_acquisition_terms(label: str, query: str, limit: int = 8) -> list[str]:
    """Generate a small set of related acquisition queries from a weak topic."""
    variants: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        variants.append(cleaned)

    add(query)
    add(label)
    add(f"{label} {query}")

    tokens = [tok.lower() for tok in re.findall(r"[a-zA-Z0-9]+", f"{label} {query}")]
    tokens = [tok for tok in tokens if len(tok) > 2 and tok not in _STOP_WORDS]
    if not tokens:
        tokens = [tok.lower() for tok in re.findall(r"[a-zA-Z0-9]+", f"{label} {query}") if tok]

    for width in (2, 3):
        for idx in range(len(tokens) - width + 1):
            add(" ".join(tokens[idx : idx + width]))
            if len(variants) >= limit:
                return variants[:limit]

    if tokens:
        add(" ".join(tokens[: min(4, len(tokens))]))
        add(" ".join(tokens[-min(4, len(tokens)) :]))

    return variants[:limit]


def _unique_sentences(sentences: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for sentence in sentences:
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def _fetch_zip_kiwix_sentences(topic: str, archive_path: str, max_sentences: int = 40) -> list[str]:
    """Fallback reader for lightweight zip-based ZIM fixtures."""
    if not zipfile.is_zipfile(archive_path):
        return []

    query_tokens = _topic_tokens(topic)
    candidates: list[tuple[str, str]] = []

    try:
        with zipfile.ZipFile(archive_path) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                lower_name = name.lower()
                try:
                    raw = zf.read(name).decode("utf-8", errors="ignore")
                except Exception:
                    continue
                text = _strip_html(raw)
                lower_text = text.lower()
                if query_tokens and not any(
                    token in lower_name or token in lower_text for token in query_tokens
                ):
                    continue
                candidates.append((name, text))
    except Exception:
        return []

    sentences: list[str] = []
    seen: set[str] = set()
    for _name, text in candidates:
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            cleaned = sentence.strip()
            if len(cleaned) <= 20:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            sentences.append(cleaned)
            if len(sentences) >= max_sentences:
                return sentences
    return sentences


def _fetch_kiwix_sentences(topic: str, max_sentences: int = 40,
                           zim_path: Optional[str] = None) -> list[str]:
    """Fetch offline content for *topic* from a local ZIM archive when available."""
    archive_path = _kiwix_archive_path(zim_path)
    if not archive_path:
        return []

    try:
        from libzim.reader import Archive
        from libzim.search import Query, Searcher
    except Exception:
        return _fetch_zip_kiwix_sentences(topic, archive_path, max_sentences=max_sentences)

    results = []
    zim = None
    try:
        zim = Archive(archive_path)
        searcher = Searcher(zim)
        query = Query().set_query(topic)
        search = searcher.search(query)
        match_count = int(getattr(search, "getEstimatedMatches", lambda: 0)() or 0)
        results = list(search.getResults(0, min(max(match_count, 1), 8)))
    except Exception:
        return _fetch_zip_kiwix_sentences(topic, archive_path, max_sentences=max_sentences)

    sentences: list[str] = []
    seen: set[str] = set()

    for result in results:
        path = getattr(result, "path", None) or getattr(result, "get_path", lambda: None)()
        if not path:
            path = str(result)
        if not path:
            continue
        normalized_path = str(path).lstrip("/")
        try:
            entry = zim.get_entry_by_path(normalized_path)
            item = entry.get_item()
            raw_content = item.content
            if isinstance(raw_content, (bytes, bytearray)):
                text = raw_content.decode("utf-8", errors="ignore")
            else:
                text = str(raw_content)
            text = _strip_html(text)
        except Exception:
            continue

        for sentence in re.split(r"(?<=[.!?])\s+", text):
            cleaned = sentence.strip()
            if len(cleaned) <= 20:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            sentences.append(cleaned)
            if len(sentences) >= max_sentences:
                return sentences

    if sentences:
        return sentences
    return _fetch_zip_kiwix_sentences(topic, archive_path, max_sentences=max_sentences)


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
                               fetched_titles: set[str] | None = None,
                               zim_path: Optional[str] = None) -> list[str]:
    """Fetch Wikipedia content for *topic*, skipping already-fetched titles.

    All article fetches run in parallel threads with a 12s total budget.
    When the primary article is already known, follows links to find new content.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

    if fetched_titles is None:
        fetched_titles = set()

    title = _search_wikipedia_title(topic)
    if not title:
        return _fetch_kiwix_sentences(topic, max_sentences=max_sentences, zim_path=zim_path)

    # Determine which titles to fetch
    if title not in fetched_titles:
        primary_titles = [title]
    else:
        related = _fetch_wikipedia_related_titles(title, limit=8)
        primary_titles = [t for t in related if t not in fetched_titles][:3]

    if not primary_titles:
        return []

    all_sentences: list[str] = []

    # Fetch all primary titles in parallel, but do not block shutdown if one hangs.
    pool = ThreadPoolExecutor(max_workers=4)
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
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    if not all_sentences:
        offline_sentences = _fetch_kiwix_sentences(title, max_sentences=max_sentences, zim_path=zim_path)
        if offline_sentences:
            all_sentences.extend(offline_sentences)

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

        pool = ThreadPoolExecutor(max_workers=2)
        link_futures = [pool.submit(_background_fetch, t) for t in link_titles[:4]]
        try:
            for fut in as_completed(link_futures, timeout=8):
                fut.result(timeout=0)
        except FuturesTimeout:
            pass
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    return all_sentences


def _fetch_dictionary_sentences(word: str) -> list[str]:
    """Fetch definitions and examples from Free Dictionary API."""
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(word)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HPM-QuizCLI/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        sentences = []
        if isinstance(data, list):
            for entry in data:
                for meaning in entry.get("meanings", []):
                    pos = meaning.get("partOfSpeech", "word")
                    for d in meaning.get("definitions", []):
                        defn = d.get("definition", "")
                        if defn:
                            sentences.append(f"The {pos} '{word}' is defined as: {defn}")
                        example = d.get("example", "")
                        if example:
                            sentences.append(f"An example of using '{word}' is: {example}")
        return sentences
    except Exception:
        return []


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
                         min_entropy: float = 0.1, fetched_titles: set[str] | None = None,
                         zim_path: Optional[str] = None) -> None:
    """Fetch Wikipedia text and dictionary definitions for each weak topic, filter by novelty, and retrain.

    Each entry is (display_label, search_query). Sentences are scored by entropy
    (how surprising they are to the model) and only novel ones are trained on.
    """
    if not weak_topics:
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

    labels = ", ".join(
        f"{label} [{query}]" if query and query != label else label
        for label, query in weak_topics
    )
    print(f"\nTriggering knowledge acquisition for: {labels}")
    all_sentences: list[str] = []

    def _acquire_topic(label: str, query: str) -> tuple[str, list[str]]:
        search_terms = _expand_acquisition_terms(label, query)
        gathered: list[str] = []

        for search_text in search_terms:
            wiki_kwargs = {"fetched_titles": fetched_titles}
            if zim_path:
                wiki_kwargs["zim_path"] = zim_path
            gathered.extend(_fetch_wikipedia_sentences(search_text, **wiki_kwargs))
            gathered.extend(_fetch_dictionary_sentences(search_text))
            gathered.extend(_fetch_wordnet_sentences(search_text))

        return label, _unique_sentences(gathered)

    pool = ThreadPoolExecutor(max_workers=min(len(weak_topics), 4))
    futures = {pool.submit(_acquire_topic, label, query): (label, query)
               for label, query in weak_topics}
    topic_results: dict[str, list[str]] = {}
    try:
        for fut in as_completed(futures, timeout=20):
            try:
                label, sentences = fut.result(timeout=0)
                topic_results[label] = sentences
            except Exception:
                label, _ = futures[fut]
                topic_results[label] = []
    except FuturesTimeout:
        for fut, (label, _query) in futures.items():
            if label not in topic_results:
                topic_results[label] = []
            if hasattr(fut, "cancel"):
                try:
                    fut.cancel()
                except Exception:
                    pass
    finally:
        if hasattr(pool, "shutdown"):
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                pool.shutdown(wait=False)

    for label, query in weak_topics:
        sentences = topic_results.get(label, [])
        search_terms = _expand_acquisition_terms(label, query)
        print(f"  Acquiring '{label}' via {len(search_terms)} query variant(s) ...", end=" ", flush=True)
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
                link_kwargs = {"fetched_titles": fetched_titles}
                if zim_path:
                    link_kwargs["zim_path"] = zim_path
                link_sentences = _fetch_wikipedia_sentences(query, **link_kwargs)
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
        reader.flush_all()
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

    # Filter out POS tags, syntactic labels, and single-char tokens
    _bad_prefixes = ("pos_", "word_", "ctx_", "sent_", "phrase_", "dep_")
    candidates = [t for t in candidates
                  if len(t) > 2
                  and not t.lower().startswith(_bad_prefixes)
                  and not t.startswith("Pos_")
                  and t.replace("_", "").isalpha()]

    if exclude:
        candidates = [t for t in candidates if t not in exclude]

    reasoning_agent._ensure_fresh()
    source_keys = reasoning_agent.iter_source_keys()

    scored: list[tuple[int, str]] = []
    for topic in candidates:
        topic_word = topic.lower().split()[0]  # use first word for lookup
        edge_count = sum(1 for k in source_keys if topic_word in k.lower())
        scored.append((edge_count, topic))

    # Ascending by edge count: fewest edges = least learned = learn this first
    scored.sort(key=lambda x: x[0])
    return [topic for _, topic in scored[:n]]


# ── KnowledgeFrontier ────────────────────────────────────────────────────────

import json as _json
from pathlib import Path as _Path

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "but", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "it", "its", "this", "that", "with",
}

_DEFAULT_FRONTIER_PATH = (
    _Path(__file__).parent.parent / "data" / "quiz_banks" / "knowledge_frontier.json"
)


class KnowledgeFrontier:
    """Progressively expands Wikipedia fetch targets via WordNet semantic graph.

    Seeds come from _nominate_uncertain_topics(); the frontier is scored by
    pager edge density so the sparsest (least-known) concepts are fetched first.
    """

    def __init__(self):
        self.known_seeds: set[str] = set()
        self.frontier: list[str] = []
        self.exhausted: set[str] = set()
        self.hop_depth: int = 1

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> None:
        target = _Path(path) if path else _DEFAULT_FRONTIER_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "known_seeds": sorted(self.known_seeds),
            "frontier": self.frontier,
            "exhausted": sorted(self.exhausted),
            "hop_depth": self.hop_depth,
        }
        target.write_text(_json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | None = None) -> "KnowledgeFrontier":
        target = _Path(path) if path else _DEFAULT_FRONTIER_PATH
        kf = cls()
        if not target.exists():
            return kf
        try:
            data = _json.loads(target.read_text())
            kf.known_seeds = set(data.get("known_seeds", []))
            kf.frontier = data.get("frontier", [])
            kf.exhausted = set(data.get("exhausted", []))
            kf.hop_depth = int(data.get("hop_depth", 1))
        except Exception:
            pass  # corrupt file → fresh state
        return kf

    # ── hop depth ────────────────────────────────────────────────────────────

    def increment_hop(self) -> None:
        self.hop_depth = min(self.hop_depth + 1, 5)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _wordnet_candidates(self, term: str, hop_depth: int | None = None) -> set[str]:
        if hop_depth is None:
            hop_depth = self.hop_depth
        try:
            from nltk.corpus import wordnet
        except ImportError:
            return set()

        synsets = wordnet.synsets(term.lower().replace(" ", "_"))[:3]
        candidates: set[str] = set()

        def _name(syn) -> str:
            return syn.lemmas()[0].name().replace("_", " ")

        for syn in synsets:
            for hop1 in syn.hypernyms() + syn.hyponyms():
                candidates.add(_name(hop1))
                if hop_depth >= 2:
                    for hop2 in hop1.hypernyms() + hop1.hyponyms():
                        candidates.add(_name(hop2))

        # Filter noise
        filtered = set()
        for c in candidates:
            if len(c) <= 1:
                continue
            if c.startswith("pos_") or c.startswith("word_"):
                continue
            if c.lower() in _STOPWORDS:
                continue
            filtered.add(c)
        return filtered

    def _edge_density(self, term: str, reader) -> int:
        word = term.lower().split()[0]
        count = 0
        for agent in reader.agents.values():
            pager = getattr(agent, "pattern_pager", None)
            if pager is None:
                continue
            for payload in pager.iter_index_payloads():
                if word in str(payload.get("name", "")).lower():
                    count += 1
        return count

    # ── public API ───────────────────────────────────────────────────────────

    def add_learned_seeds(self, terms: list[str], reader) -> None:
        """Expand frontier from new seed terms via WordNet."""
        for term in terms:
            key = term.lower().strip()
            if key in self.known_seeds:
                continue
            self.known_seeds.add(key)
            candidates = self._wordnet_candidates(key, self.hop_depth)
            # Remove already known / exhausted
            candidates -= self.known_seeds
            candidates -= self.exhausted
            # Remove terms already in frontier
            existing = set(self.frontier)
            candidates -= existing
            # Score by edge density (ascending = most to learn)
            scored = sorted(candidates, key=lambda c: self._edge_density(c, reader))
            self.frontier.extend(scored[:8])

    def next_topics(self, reader, n: int = 4) -> list[str]:
        """Return up to n frontier topics with the lowest edge density."""
        if not self.frontier:
            return []
        # Re-score frontier live
        scored = sorted(self.frontier, key=lambda c: self._edge_density(c, reader))
        chosen = scored[:n]
        # Remove chosen from frontier
        chosen_set = set(chosen)
        self.frontier = [t for t in self.frontier if t not in chosen_set]
        return chosen


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    global _KIWIX_ZIM_OVERRIDE
    _KIWIX_ZIM_OVERRIDE = getattr(args, "kiwix_zim_path", None)

    print("Building HPM reader...")
    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
    from hpm_ai_v6.agents.quiz_agent import QuizAgent
    from hpm_ai_v6.agents.dataset_training_agent import DatasetTrainingAgent

    corpus = _corpus_path()
    reader = MultiAgentReader(corpus, warm_start=False)  # on-demand retrieval from pager index
    reasoning_agent = getattr(reader, "reasoning_agent", None)
    if reasoning_agent is None:
        print(red("Error: reasoning_agent not found on MultiAgentReader."))
        sys.exit(1)

    quiz_agent = QuizAgent(reader, reasoning_agent)

    mastered: set[str] = set()
    round_num = 0

    def _learn_from_attempt(payload: dict) -> None:
        learn_from_quiz_attempt(reader, payload)

    while True:
        round_num += 1
        if args.loop and round_num > 1:
            print(f"\n{yellow(f'=== Loop round {round_num} ===')} (mastered {len(mastered)} question(s) so far)")

        _score, _weak, newly_mastered = run_quiz(
            reader=reader,
            quiz_agent=quiz_agent,
            reasoning_agent=reasoning_agent,
            n=args.n,
            difficulty=args.difficulty,
            source=args.source,
            auto=args.auto,
            skip_ids=mastered,
            feedback_hook=_learn_from_attempt,
        )
        mastered.update(newly_mastered)
        reader.flush_all()  # persist newly trained patterns to SQLite

        if not args.loop:
            break

        remaining = args.n - len(mastered)
        if remaining <= 0:
            print(green(f"\nAll {args.n} questions mastered! Quiz complete."))
            break
        print(f"\n{remaining} question(s) still to master. Rerunning...")


def _persist_all_patterns(reader) -> int:
    """Enqueue all in-memory patterns from pager-enabled agents to SQLite.

    Called after training so patterns don't require eviction to reach the DB.
    Returns total count saved.
    """
    total = 0
    for agent in reader.agents.values():
        pager = getattr(agent, "pattern_pager", None)
        if pager is None:
            continue
        for pattern in getattr(agent, "patterns", []):
            try:
                pager.enqueue_save(pattern)
                total += 1
            except Exception:
                pass
    return total


OPTION_KEYS = ["A", "B", "C", "D"]


def _retrieve_relevant_patterns(reader, question: str, options_map: dict, top_k: int = 50) -> int:
    """Load patterns relevant to question terms and options from each agent's archive.

    Uses each agent's hydrate_patterns_from_archive with term-based pre-filtering
    so that source/target Cell objects are correctly reconstructed via the agent's
    own cell registry (populated by the minimal warm_start).
    """
    terms = {w.lower() for w in question.split() if w.isalnum() and len(w) > 1 and w.lower() not in _STOP_WORDS}
    for opt in options_map.values():
        terms.update(_tok(w) for w in (opt or "").split() if _tok(w) and len(_tok(w)) > 1)

    total_loaded = 0
    for agent in reader.agents.values():
        pager = getattr(agent, "pattern_pager", None)
        if pager is None or not hasattr(agent, "hydrate_patterns_from_archive"):
            continue
        existing = {p.name for p in getattr(agent, "patterns", [])}
        payloads = pager.iter_index_payloads()
        # Filter to relevant names not already loaded
        relevant = sorted(
            [p for p in payloads
             if any(t in str(p.get("name", "")).lower() for t in terms)
             and str(p.get("name", "")) not in existing],
            key=lambda p: float(p.get("weight", 0.0)),
            reverse=True,
        )[:top_k]

        if not relevant:
            continue

        lookup = agent._paging_lookup()
        loaded = 0
        for payload in relevant:
            try:
                pattern = pager.load_from_payload(payload, lookup)
                agent.patterns.append(pattern)
                loaded += 1
            except Exception:
                continue

        if loaded > 0 and hasattr(agent, "_refresh_learner"):
            agent._refresh_learner()
        total_loaded += loaded

    return total_loaded


def _score_options(reader, question: str, options_map: dict) -> tuple[str, bool, dict]:
    """Score each option against patterns from both active memory and the pager index.

    Checks agent.patterns (in-memory, includes just-trained) and pager index (persisted).
    Options whose words appear in more/heavier patterns score higher.
    """
    direct_key = _lookup_quiz_memory(reader, question, options_map)
    if direct_key:
        return direct_key, True, {
            "direct_memory": True,
            "scores": {k: (1.0 if k == direct_key else 0.0) for k in options_map},
        }

    q_words = {_tok(w) for w in question.split() if _tok(w) not in _STOP_WORDS and len(_tok(w)) > 1}

    scores: dict[str, float] = {}
    for key, option_text in options_map.items():
        if not option_text:
            scores[key] = 0.0
            continue
        opt_words = [_tok(w) for w in option_text.split() if _tok(w)]
        total = 0.0
        seen_names: set[str] = set()

        for agent in reader.agents.values():
            # Check in-memory active patterns first (includes just-trained corrections)
            for pattern in getattr(agent, "patterns", []):
                name = str(pattern.name).lower()
                if name in seen_names:
                    continue
                seen_names.add(name)
                weight = float(getattr(pattern, "weight", 1.0))
                has_opt = any(w in name for w in opt_words)
                has_q = any(qw in name for qw in q_words)
                if has_opt and has_q:
                    total += weight

            # Also check pager index (evicted/persisted patterns)
            pager = getattr(agent, "pattern_pager", None)
            if pager is None:
                continue
            for payload in pager.iter_index_payloads():
                name = str(payload.get("name", "")).lower()
                if name in seen_names:
                    continue
                seen_names.add(name)
                weight = float(payload.get("weight", 0.0))
                has_opt = any(w in name for w in opt_words)
                has_q = any(qw in name for qw in q_words)
                if has_opt and has_q:
                    total += weight

        scores[key] = total

    best_key = max(scores, key=lambda k: scores[k])
    best_score = scores[best_key]
    confident = best_score > 0.0
    if not confident:
        return "", False, {"scores": scores}
    return best_key, True, {"scores": scores}


def _normalize_quiz_text(text: str) -> str:
    """Normalize a question or answer into a stable lookup key."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _quiz_memory_name(question: str, correct_key: str) -> str:
    """Build a direct-memory key for an exact question->answer association."""
    return f"quiz_answer::{_normalize_quiz_text(question)}=>{correct_key.lower()}"


def _lookup_quiz_memory(reader, question: str, options_map: dict) -> str:
    """Return a directly learned answer key for an exact quiz question, if present."""
    question_key = _normalize_quiz_text(question)
    if not question_key:
        return ""

    prefix = f"quiz_answer::{question_key}=>"
    best_key = ""
    best_weight = 0.0

    for agent in reader.agents.values():
        for pattern in getattr(agent, "patterns", []):
            name = str(getattr(pattern, "name", "")).lower()
            if not name.startswith(prefix):
                continue
            key = name[len(prefix):].upper()
            if key not in options_map:
                continue
            weight = float(getattr(pattern, "weight", 0.0))
            if weight > best_weight:
                best_key = key
                best_weight = weight

        pager = getattr(agent, "pattern_pager", None)
        if pager is None:
            continue
        for payload in pager.iter_index_payloads():
            name = str(payload.get("name", "")).lower()
            if not name.startswith(prefix):
                continue
            key = name[len(prefix):].upper()
            if key not in options_map:
                continue
            weight = float(payload.get("weight", 0.0))
            if weight > best_weight:
                best_key = key
                best_weight = weight

    return best_key


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
    return ""


def _extract_answer(trace: dict, options: dict) -> str:
    """Map reasoning trace output to an option key A/B/C/D.

    Returns an empty string when no confident option can be inferred.
    """
    # If the agent set an explicit 'answer' key, trust it.
    answer_text = str(trace.get("answer", "") or "")
    if answer_text in OPTION_KEYS:
        return answer_text
    parsed = _parse_letter_answer(answer_text, options)
    if parsed in OPTION_KEYS:
        return parsed

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

    return ""


_STOP_WORDS = {"what", "is", "the", "a", "an", "of", "in", "on", "at", "to", "for",
               "are", "was", "were", "does", "do", "how", "many", "which", "who",
               "where", "when", "why", "that", "this", "it", "its", "have", "has"}

def _tok(word: str) -> str:
    """Strip punctuation, return lowercase alphanumeric token."""
    return re.sub(r"[^\w]", "", word).lower()

def _build_search_query(question: str, correct_answer: str) -> str:
    """Build a specific Wikipedia search query from question keywords + correct answer."""
    # Extract meaningful words from the question (drop stop words and punctuation)
    words = re.sub(r"[^\w\s]", " ", question.lower()).split()
    keywords = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    # Combine up to 4 question keywords with the correct answer
    query_parts = keywords[:4] + [correct_answer]
    return " ".join(query_parts)


def _boost_correct_patterns(reader, question: str, correct_text: str, boost: float = 20.0) -> int:
    """Boost Cell.weight for patterns encoding question→correct_answer associations.

    Works on all agents regardless of whether they have a meta_rule.
    Boosted weights are serialized to SQLite on next flush_all(), persisting
    the stronger association across sessions.

    Returns the number of patterns boosted.
    """
    q_words = {w for w in re.sub(r"[^\w\s]", " ", question.lower()).split()
               if w not in _STOP_WORDS and len(w) > 2}
    ans_words = [_tok(w) for w in correct_text.split() if _tok(w)]
    if not q_words or not ans_words:
        return 0

    total_boosted = 0

    # Find the word agent — it has a pager and can persist associations
    word_agent = None
    for agent in reader.agents.values():
        if getattr(agent, "pattern_pager", None) is not None and hasattr(agent, "patterns"):
            if any("w_" in str(getattr(p, "name", "")) for p in getattr(agent, "patterns", [])):
                word_agent = agent
                break

    # Boost in-memory patterns across all agents (affects within-session scoring)
    for agent in reader.agents.values():
        patterns = getattr(agent, "patterns", [])
        if not patterns:
            continue
        pager = getattr(agent, "pattern_pager", None)
        for pattern in patterns:
            name = pattern.name.lower()
            has_ans = any(w in name for w in ans_words)
            has_q = any(qw in name for qw in q_words)
            if has_ans and has_q:
                pattern.weight = float(getattr(pattern, "weight", 1.0)) * boost
                total_boosted += 1
                if pager is not None:
                    pager.enqueue_save(pattern)

    # Persist direct answer→question_keyword cells into word agent pager
    # so cross-session scoring finds them even when dependency agent is gone
    if word_agent is not None:
        pager = word_agent.pattern_pager
        for ans_word in ans_words:
            for q_word in q_words:
                cell_name = f"w_{ans_word}->{q_word}"
                emb = np.zeros(16, dtype=float)
                for i, ch in enumerate((ans_word + q_word)[:16]):
                    emb[i] = ord(ch) / 128.0
                cell = Cell(name=cell_name, dim=1,
                            embedding=emb, weight=float(boost))
                pager.enqueue_save(cell)
                total_boosted += 1

    return total_boosted


def _persist_quiz_answer(reader, question: str, correct_key: str, boost: float = 100.0) -> int:
    """Persist a direct question->answer memory for exact repeated quiz items."""
    if not question or not correct_key:
        return 0
    memory_name = _quiz_memory_name(question, correct_key)
    cell = Cell(name=memory_name, dim=1, embedding=np.zeros(16, dtype=float), weight=float(boost))
    saved = 0
    for agent in reader.agents.values():
        pager = getattr(agent, "pattern_pager", None)
        if pager is None:
            continue
        try:
            pager.enqueue_save(cell)
            saved += 1
        except Exception:
            pass
    return saved


def _build_correction_sentence(question: str, correct_text: str, topic: str | None = None) -> str:
    """Create a direct supervised learning sentence from a quiz correction."""
    parts = []
    if topic:
        parts.append(f"Topic: {topic}.")
    parts.append(f"Question: {question}.")
    if correct_text:
        parts.append(f"Correct answer: {correct_text}.")
    return " ".join(parts)


def _reinforce_trace(reasoning_agent, trace: dict, boost: bool) -> None:
    """No-op: pager-based scoring doesn't use reasoning graph edges."""
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
    feedback_hook: Optional[Callable[[dict], None]] = None,
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

        chosen, confident, _ = _score_options(reader, q.question, options_map)
        trace = {}
        if not chosen:
            chosen = "?"
        
        correct = OPTION_KEYS[q.correct_index]

        confidence_label = "confident" if confident else yellow("guessing")
        print(f"\nAI answers: {chosen}  [{confidence_label}]")
        print(f"Correct answer: {correct}) {options_map.get(correct, '')}")

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

        if feedback_hook is not None:
            try:
                feedback_hook({
                    "question_id": getattr(q, "id", None),
                    "question": q.question,
                    "topic": getattr(q, "topic", None),
                    "query": _build_search_query(q.question, options_map.get(correct, "")),
                    "chosen": chosen,
                    "correct": correct,
                    "correct_text": options_map.get(correct, ""),
                    "chosen_text": options_map.get(chosen, ""),
                    "options": options_map,
                    "is_correct": is_correct,
                    "confident": confident,
                    "trace": trace,
                })
            except Exception:
                pass

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
