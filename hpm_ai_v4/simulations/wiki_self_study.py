"""Wikipedia self-study crawler for building an HPM library."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from hpm_ai_v4.evaluators.metrics import epistemic_score
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.ingest import TextIngestGate
from hpm_ai_v4.tools.dictionary import DictionaryValidator, NLTKWordList
from hpm_ai_v4.tools.grammar import GrammarValidator, HeuristicGrammarLibrary
from hpm_ai_v4.tools.serializer import PatternSerializer


WIKI_API = "https://en.wikipedia.org/w/api.php"


@dataclass(frozen=True)
class WikiPage:
    title: str
    text: str
    links: List[str] = field(default_factory=list)
    source_url: str = ""


@dataclass
class StudyResult:
    pages_read: int
    patterns_saved: int
    output_path: str
    visited: int
    queued: int


class WikiFetcher:
    """Fetch Wikipedia page extracts and outbound links via the MediaWiki API."""

    def __init__(self, api_url: str = WIKI_API, timeout: float = 20.0, user_agent: str = "HPM-Learning-Agent/1.0"):
        self.api_url = api_url.rstrip("?")
        self.timeout = float(timeout)
        self.user_agent = user_agent

    def _request_json(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = urllib.parse.urlencode(params)
        url = f"{self.api_url}?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:  # nosec: trusted API endpoint
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def canonicalize_title(title: str) -> str:
        text = str(title or "").strip()
        text = text.replace(" ", "_")
        text = re.sub(r"_+", "_", text)
        return text

    @staticmethod
    def _is_mainspace_title(title: str) -> bool:
        return bool(title) and ":" not in title and not title.startswith("#")

    def fetch(self, title: str, max_links: int = 200) -> WikiPage:
        canonical = self.canonicalize_title(title)
        params = {
            "action": "query",
            "format": "json",
            "redirects": 1,
            "prop": "extracts|links",
            "explaintext": 1,
            "exsectionformat": "plain",
            "plnamespace": 0,
            "pllimit": "max",
            "titles": canonical,
        }

        text = ""
        links: List[str] = []
        source_url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
        continuation: Dict[str, Any] = {}

        while True:
            payload = self._request_json({**params, **continuation})
            query = payload.get("query", {})
            pages = query.get("pages", {})
            page = next(iter(pages.values()), {}) if isinstance(pages, dict) else {}
            if not text:
                text = str(page.get("extract", "") or "")
                if not canonical or canonical == title:
                    canonical = str(page.get("title", canonical) or canonical)
            for link in page.get("links", []) or []:
                link_title = str(link.get("title", "") or "").strip()
                if self._is_mainspace_title(link_title):
                    links.append(link_title)
                    if len(links) >= max_links:
                        break
            if len(links) >= max_links:
                break
            cont = payload.get("continue")
            if not cont:
                break
            continuation = {k: v for k, v in cont.items() if k != "continue"}

        links = list(dict.fromkeys(links))
        return WikiPage(title=canonical, text=text, links=links, source_url=source_url)


class CuriosityScheduler:
    """Priority queue over Wikipedia titles guided by epistemic uncertainty."""

    def __init__(self, agent: Any):
        self.agent = agent
        self.queue: List[Tuple[float, str]] = []
        self.visited: set[str] = set()
        self.in_frontier: set[str] = set()
        self.page_bonus: Dict[str, float] = {}

    @staticmethod
    def _normalize_title(title: str) -> str:
        return WikiFetcher.canonicalize_title(title).strip()

    def _surface_ids(self, title: str) -> List[int]:
        if hasattr(self.agent, "_surface_ids_from_text"):
            try:
                return [int(v) for v in self.agent._surface_ids_from_text(title)]
            except Exception:
                pass
        return [max(0, min(94, ord(ch) - 32)) for ch in title if 32 <= ord(ch) <= 126]

    def score_link(self, title: str, *, source_title: str = "") -> float:
        norm = self._normalize_title(title)
        ids = self._surface_ids(norm)
        if not ids:
            return 0.0

        reasoner = getattr(self.agent, "reasoner", None)
        relevant = []
        if reasoner is not None and hasattr(reasoner, "get_relevant_patterns"):
            try:
                relevant = reasoner.get_relevant_patterns(ids, top_k=3) or []
            except Exception:
                relevant = []
        if not relevant:
            score = 1.0
        else:
            scores = [float(epistemic_score(p)) for p in relevant]
            score = 1.0 - float(np.mean(scores))

        if source_title and source_title != norm:
            score += 0.05
        score += float(self.page_bonus.get(norm, 0.0))
        return float(score)

    def push(self, links: Iterable[str], *, source_title: str = "") -> None:
        for title in links:
            norm = self._normalize_title(title)
            if not norm or norm in self.visited or norm in self.in_frontier:
                continue
            priority = self.score_link(norm, source_title=source_title)
            self.queue.append((priority, norm))
            self.in_frontier.add(norm)
        self.queue.sort(key=lambda item: (item[0], item[1]), reverse=True)

    def pop(self) -> Optional[str]:
        while self.queue:
            _, title = self.queue.pop(0)
            self.in_frontier.discard(title)
            if title in self.visited:
                continue
            self.visited.add(title)
            return title
        return None

    def seed(self, titles: Iterable[str]) -> None:
        self.push(titles)


class SelfStudyAgent:
    """Fetch Wikipedia pages, train on them, and follow links by curiosity."""

    def __init__(
        self,
        seed_topics: Sequence[str],
        output_library: str,
        *,
        steps_per_chunk: int = 500,
        max_pages: int = 500,
        target_patterns: int = 2000,
        max_links_per_page: int = 50,
        chunk_char_budget: int = 1800,
        surface_mode: str = "word",
        agent: Optional[LayeredAgent] = None,
        fetcher: Optional[WikiFetcher] = None,
        scheduler: Optional[CuriosityScheduler] = None,
        dictionary: Optional[DictionaryValidator] = None,
        grammar: Optional[GrammarValidator] = None,
        sentence_features: bool = True,
        resume_library: Optional[str] = None,
    ):
        self.output_library = output_library
        self.steps_per_chunk = max(1, int(steps_per_chunk))
        self.max_pages = max(1, int(max_pages))
        self.target_patterns = max(1, int(target_patterns))
        self.max_links_per_page = max(1, int(max_links_per_page))
        self.chunk_char_budget = max(200, int(chunk_char_budget))
        self.fetcher = fetcher or WikiFetcher()
        self.agent = agent or LayeredAgent(
            num_workers=1,
            dictionary=dictionary,
            grammar=grammar or HeuristicGrammarLibrary(),
            surface_mode=surface_mode,
        )
        self.scheduler = scheduler or CuriosityScheduler(self.agent)
        self._seed_topics = [self.fetcher.canonicalize_title(topic) for topic in seed_topics if str(topic).strip()]
        self.scheduler.seed(self._seed_topics)
        self._read_pages: List[str] = []
        self._sentence_features = bool(sentence_features)
        ingest_path = self._ingest_path(self.output_library)
        self._ingest_gate = TextIngestGate.load_snapshot_from_path(
            ingest_path,
            adapter=getattr(self.agent, "_adapter", None),
            lowercase=bool(getattr(getattr(self.agent, "_adapter", None), "lowercase", True)),
        )
        # Auto-resume: reload previously learned patterns from resume_library or output_library
        _resume = resume_library if resume_library is not None else self.output_library
        if _resume and os.path.exists(_resume):
            self._load_library(_resume)

    @staticmethod
    def _ingest_path(path: str) -> str:
        return path[:-4] + ".ingest.json" if path.endswith(".pkl") else path + ".ingest.json"

    def _training_patterns(self) -> List[Any]:
        patterns = getattr(self.agent, "patterns", None)
        if patterns is not None:
            return list(patterns)
        if hasattr(self.agent, "l1") and getattr(self.agent.l1, "patterns", None) is not None:
            return list(self.agent.l1.patterns)
        return []

    def _load_library(self, path: str) -> int:
        """Load patterns from a previously saved library into the agent. Returns count loaded."""
        if not os.path.exists(path):
            return 0
        try:
            patterns = PatternSerializer.load(path)
            if hasattr(self.agent, "l1"):
                self.agent.l1.patterns = patterns
            elif hasattr(self.agent, "patterns"):
                self.agent.patterns = patterns
            print(f"[resume] loaded {len(patterns)} patterns from {path}", flush=True)
            return len(patterns)
        except Exception as exc:
            print(f"[resume] failed to load {path}: {exc}", flush=True)
            return 0

    def _save_library(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if hasattr(self.agent, "save_bundle"):
            base = path[:-4] if path.endswith(".pkl") else path
            self.agent.save_bundle(base)
            ingest_path = self._ingest_path(base)
            with open(ingest_path, "w", encoding="utf-8") as f:
                json.dump(self._ingest_gate.snapshot(), f, sort_keys=True)
            return
        PatternSerializer.save(self._training_patterns(), path)
        ingest_path = self._ingest_path(path)
        with open(ingest_path, "w", encoding="utf-8") as f:
            json.dump(self._ingest_gate.snapshot(), f, sort_keys=True)

    def _chunk_page_text(self, text: str) -> List[str]:
        clean = str(text or "").strip()
        if not clean:
            return []

        # Prefer sentence-boundary chunking if available; otherwise fall back to character windows.
        try:
            from hpm_ai_v4.io.adapters import SentenceAdapter

            adapter = SentenceAdapter(use_spacy=False)
            sentences = [span.text.strip() for span in adapter.segment(clean) if span.text.strip()]
        except Exception:
            sentences = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", clean) if piece.strip()]

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for sentence in sentences:
            sentence_len = len(sentence)
            if current and current_len + sentence_len > self.chunk_char_budget:
                chunks.append(" ".join(current).strip())
                current = [sentence]
                current_len = sentence_len
            else:
                current.append(sentence)
                current_len += sentence_len + 1
        if current:
            chunks.append(" ".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _split_chunk_windows(self, chunk: str) -> List[str]:
        clean = str(chunk or "").strip()
        if not clean:
            return []
        try:
            from hpm_ai_v4.io.adapters import SentenceAdapter

            adapter = SentenceAdapter(use_spacy=False)
            windows = [span.text.strip() for span in adapter.segment(clean) if span.text.strip()]
        except Exception:
            windows = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", clean) if piece.strip()]
        if len(windows) <= 1:
            return [clean]
        return windows

    def _chunk_already_covered(self, tokens: List[int], threshold: float = -0.35, top_k: int = 5) -> bool:
        """Return True if existing patterns already model this token sequence well.

        Scores the top-K weighted hierarchical patterns against the token window.
        If the best log-likelihood per token exceeds threshold, the chunk is
        already represented in the library — skip training.
        """
        if not tokens or len(tokens) < 4:
            return False
        patterns = getattr(self.agent, "patterns", None)
        if not patterns:
            return False
        hier = sorted(
            [p for p in patterns if getattr(p, "latent_dim", 0) > 1],
            key=lambda p: float(p.weight),
            reverse=True,
        )[:top_k]
        if not hier:
            return False
        sample = tokens[:min(60, len(tokens))]
        best_ll = float("-inf")
        for p in hier:
            try:
                ll = float(p.log_likelihood(sample)) / max(1, len(sample))
                if ll > best_ll:
                    best_ll = ll
            except Exception:
                continue
        return best_ll >= threshold

    def _warm_start_chunk(self, tokens: List[int], boost: float = 0.15, top_k: int = 5) -> None:
        """Boost weights of existing patterns most relevant to this token window.

        Patterns that already encode structure relevant to the incoming chunk
        get a weight nudge before training begins, so learning builds on prior
        knowledge rather than re-discovering it from scratch.
        """
        if not hasattr(self.agent, "reasoner") or not hasattr(self.agent, "patterns"):
            return
        try:
            relevant = self.agent.reasoner.get_relevant_patterns(tokens, top_k=top_k)
            if not relevant:
                return
            total_w = sum(float(p.weight) for p in self.agent.patterns) + 1e-12
            for p in relevant:
                p.weight = float(np.clip(p.weight + boost / max(1, len(relevant)), 0.0, 1.0))
            # Re-normalise so total weight stays stable
            new_total = sum(float(p.weight) for p in self.agent.patterns) + 1e-12
            scale = total_w / new_total
            for p in self.agent.patterns:
                p.weight = float(np.clip(p.weight * scale, 1e-6, 1.0))
        except Exception:
            pass

    def _consolidate_patterns(self, sim_threshold: float = 0.97) -> int:
        """Remove near-duplicate patterns after a page training pass."""
        patterns = getattr(self.agent, "patterns", None)
        if not patterns or len(patterns) < 10:
            return 0
        try:
            from hpm_ai_v4.simulations.build_large_nlp_library import deduplicate
            before = len(patterns)
            deduped = deduplicate(patterns, sim_threshold=sim_threshold)
            self.agent.patterns = deduped
            return before - len(deduped)
        except Exception:
            return 0

    def _train_on_page(self, page: WikiPage) -> Dict[str, Any]:
        chunks = self._chunk_page_text(page.text)
        if not chunks:
            return {"chunks": 0, "chars": 0}

        learned = 0
        chars = 0
        windows_seen = 0
        windows_novel = 0
        selected_chunks = chunks[: self.steps_per_chunk]
        total_chunks = len(selected_chunks)
        for idx, chunk in enumerate(selected_chunks, start=1):
            if not self._ingest_gate.register_text(chunk):
                if idx == 1 or idx == total_chunks or idx % 10 == 0:
                    print(
                        f"[train] {page.title} chunk {idx}/{total_chunks} skipped duplicate",
                        flush=True,
                    )
                continue
            chunk_windows = self._split_chunk_windows(chunk)
            novel_windows: List[str] = []
            near_windows = 0
            for window in chunk_windows:
                windows_seen += 1
                state, score = self._ingest_gate.match_window(window)
                if state == "known":
                    continue
                if state == "near":
                    near_windows += 1
                    self._ingest_gate.near_duplicate_windows += 1
                    continue
                if self._ingest_gate.register_window(window):
                    novel_windows.append(window)
                    windows_novel += 1
            if not novel_windows:
                if idx == 1 or idx == total_chunks or idx % 10 == 0:
                    print(
                        f"[train] {page.title} chunk {idx}/{total_chunks} skipped window-reuse",
                        flush=True,
                    )
                continue
            chunk_start = time.perf_counter()
            adapter = getattr(self.agent, "_adapter", None)
            stats = {"target_chars": 0, "self_chars": 0, "token_agreement": 0.0, "plausibility": 0.0}
            covered_windows = 0
            filtered_windows: List[str] = []
            for window in novel_windows:
                if adapter is not None:
                    try:
                        wtokens = [adapter.encode_char(c) for c in window
                                   if 32 <= ord(c) <= 127 or c == '\n']
                        if self._chunk_already_covered(wtokens):
                            covered_windows += 1
                            continue
                        self._warm_start_chunk(wtokens)
                    except Exception:
                        pass
                filtered_windows.append(window)
            if not filtered_windows:
                if idx == 1 or idx == total_chunks or idx % 10 == 0:
                    print(
                        f"[train] {page.title} chunk {idx}/{total_chunks} "
                        f"skipped pattern-covered ({covered_windows} windows)",
                        flush=True,
                    )
                continue
            novel_windows = filtered_windows
            for window in novel_windows:
                window_stats = self.agent.observe_text(
                    window,
                    feedback_mode="target",
                    self_feedback_weight=0.0,
                )
                stats["target_chars"] += int(window_stats.get("target_chars", 0))
                stats["self_chars"] += int(window_stats.get("self_chars", 0))
                stats["token_agreement"] = max(float(stats["token_agreement"]), float(window_stats.get("token_agreement", 0.0)))
                stats["plausibility"] = max(float(stats["plausibility"]), float(window_stats.get("plausibility", 0.0)))
            chunk_elapsed = time.perf_counter() - chunk_start
            learned += int(stats.get("target_chars", 0))
            chars += sum(len(window) for window in novel_windows)
            if idx == 1 or idx == total_chunks or idx % 10 == 0:
                print(
                    f"[train] {page.title} chunk {idx}/{total_chunks} "
                    f"windows={len(novel_windows)}/{len(chunk_windows)} "
                    f"near={near_windows} "
                    f"chars={sum(len(window) for window in novel_windows)} learned={int(stats.get('target_chars', 0))} "
                    f"elapsed={chunk_elapsed:.2f}s",
                    flush=True,
                )
        removed = self._consolidate_patterns()
        if removed:
            print(f"[consolidate] {page.title} removed {removed} near-duplicate patterns", flush=True)
        return {"chunks": len(chunks), "chars": chars, "learned": learned, "windows_seen": windows_seen, "windows_novel": windows_novel}

    def _fetch_page(self, title: str):
        """Fetch a single page; returns (page, elapsed) or (None, elapsed) on error."""
        t = time.perf_counter()
        try:
            page = self.fetcher.fetch(title, max_links=self.max_links_per_page)
            return page, time.perf_counter() - t
        except Exception as exc:
            print(f"[skip] {title}: {exc}", flush=True)
            return None, time.perf_counter() - t

    def study(self) -> StudyResult:
        from concurrent.futures import ThreadPoolExecutor, Future
        pages_read = 0
        prefetch_future: Future | None = None
        prefetch_title: str | None = None

        executor = ThreadPoolExecutor(max_workers=1)

        def _next_fetch():
            t = self.scheduler.pop()
            if t is None:
                return None, None
            return t, executor.submit(self._fetch_page, t)

        # Kick off first prefetch
        prefetch_title, prefetch_future = _next_fetch()

        while pages_read < self.max_pages:
            if prefetch_future is None:
                break

            # Collect current page result
            title = prefetch_title
            page, fetch_elapsed = prefetch_future.result()

            if page is None or not page.text.strip():
                # Still need to advance the prefetch
                prefetch_title, prefetch_future = _next_fetch()
                continue

            # Push links before kicking off next prefetch so single-seed runs work
            self.scheduler.push(page.links[: self.max_links_per_page], source_title=page.title)

            # Immediately kick off next prefetch while we train
            prefetch_title, prefetch_future = _next_fetch()

            self._read_pages.append(page.title)
            train_start = time.perf_counter()
            train_stats = self._train_on_page(page)
            train_elapsed = time.perf_counter() - train_start
            pages_read += 1

            pattern_count = len([p for p in self._training_patterns() if getattr(p, "latent_dim", 0) > 1])
            print(
                f"[{pages_read}] {page.title} — {pattern_count} patterns "
                f"(fetch={fetch_elapsed:.2f}s train={train_elapsed:.2f}s "
                f"chunks={train_stats.get('chunks', 0)} chars={train_stats.get('chars', 0)})",
                flush=True,
            )

            self.scheduler.page_bonus[page.title] = max(
                self.scheduler.page_bonus.get(page.title, 0.0) * 0.9,
                0.05,
            )

            if pages_read % 50 == 0:
                ckpt_path = self.output_library.replace(".pkl", f"_ckpt{pages_read}.pkl")
                print(f"[checkpoint] saving {ckpt_path}", flush=True)
                self._save_library(ckpt_path)

            if pattern_count >= self.target_patterns:
                print(f"[done] target reached at page {pages_read}", flush=True)
                break

        executor.shutdown(wait=False)

        print(
            f"[done] study complete pages={pages_read} visited={len(self.scheduler.visited)} "
            f"queued={len(self.scheduler.queue)}",
            flush=True,
        )
        self._save_library(self.output_library)
        return StudyResult(
            pages_read=pages_read,
            patterns_saved=len(self._training_patterns()),
            output_path=self.output_library,
            visited=len(self.scheduler.visited),
            queued=len(self.scheduler.queue),
        )


def build_wikipedia_self_study(
    seed_topics: Sequence[str],
    output_library: str,
    *,
    steps_per_chunk: int = 500,
    max_pages: int = 500,
    target_patterns: int = 2000,
    max_links_per_page: int = 50,
    chunk_char_budget: int = 1800,
    surface_mode: str = "word",
    dictionary: Optional[DictionaryValidator] = None,
    grammar: Optional[GrammarValidator] = None,
    resume_library: Optional[str] = None,
) -> StudyResult:
    agent = SelfStudyAgent(
        seed_topics=seed_topics,
        output_library=output_library,
        steps_per_chunk=steps_per_chunk,
        max_pages=max_pages,
        target_patterns=target_patterns,
        max_links_per_page=max_links_per_page,
        chunk_char_budget=chunk_char_budget,
        surface_mode=surface_mode,
        dictionary=dictionary,
        grammar=grammar,
        resume_library=resume_library,
    )
    return agent.study()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wikipedia self-study crawler for HPM")
    parser.add_argument("--seed", nargs="+", required=True, help="Seed Wikipedia topics")
    _default_library = os.path.join(os.path.expanduser("~"), ".hpm", "library.pkl")
    parser.add_argument("--output", default=_default_library, help="Output library path (.pkl base path)")
    parser.add_argument("--resume", default=None, help="Load patterns from this library before starting (defaults to --output if it exists)")
    parser.add_argument("--steps-per-chunk", type=int, default=500, help="Chunk budget used during reading")
    parser.add_argument("--max-pages", type=int, default=500, help="Maximum pages to read")
    parser.add_argument("--target-patterns", type=int, default=2000, help="Stop once this many hierarchical patterns are retained")
    parser.add_argument("--max-links-per-page", type=int, default=50, help="Maximum outbound links to queue per page")
    parser.add_argument("--chunk-char-budget", type=int, default=1800, help="Maximum characters per training chunk")
    parser.add_argument("--surface-mode", default="word", help="LayeredAgent surface mode")
    parser.add_argument("--dict", action="store_true", help="Attach an NLTK dictionary and heuristic grammar")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dictionary = NLTKWordList(download=False) if args.dict else None
    grammar = HeuristicGrammarLibrary() if args.dict else None
    result = build_wikipedia_self_study(
        seed_topics=args.seed,
        output_library=args.output,
        steps_per_chunk=args.steps_per_chunk,
        max_pages=args.max_pages,
        target_patterns=args.target_patterns,
        max_links_per_page=args.max_links_per_page,
        chunk_char_budget=args.chunk_char_budget,
        surface_mode=args.surface_mode,
        dictionary=dictionary,
        grammar=grammar,
        resume_library=args.resume,
    )
    print(f"[done] pages={result.pages_read} patterns={result.patterns_saved} output={result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
