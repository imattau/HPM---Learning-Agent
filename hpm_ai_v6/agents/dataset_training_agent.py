#!/usr/bin/env python3
"""
Dataset training agent for the V6 HPM multi-agent reader.

Loads datasets from Hugging Face or local text files, scores examples using
the current contextual model, appends the most informative examples to the
reader corpus, and triggers incremental retraining.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import requests

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional runtime dependency
    load_dataset = None


class DatasetTrainingAgent:
    CURATED_GUTENBERG_BOOK_IDS = (
        11,    # Alice's Adventures in Wonderland
        84,    # Frankenstein
        1342,  # Pride and Prejudice
        76,    # Adventures of Huckleberry Finn
        98,    # A Tale of Two Cities
        2701,  # Moby Dick
        1661,  # The Adventures of Sherlock Holmes
        345,   # Dracula
        2600,  # War and Peace
        5200,  # Metamorphosis
        1260,  # Jane Eyre
    )

    def __init__(
        self,
        multi_agent_reader,
        corpus_path: str,
        min_sentence_len: int = 20,
        context_len: int = 3,
    ):
        self.reader = multi_agent_reader
        self.corpus_path = corpus_path
        self.min_len = min_sentence_len
        self.context_len = context_len
        self.contextual_agent = self.reader.agents.get("contextual")
        if self.contextual_agent is None:
            raise ValueError("MultiAgentReader must expose a 'contextual' agent.")

    @classmethod
    def curated_gutenberg_book_ids(cls) -> List[int]:
        return list(cls.CURATED_GUTENBERG_BOOK_IDS)

    @staticmethod
    def _gutenberg_url(book_id: int) -> str:
        return f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

    @staticmethod
    def _strip_gutenberg_boilerplate(text: str) -> str:
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG EBOOK",
            "*** START OF THIS PROJECT GUTENBERG EBOOK",
        ]
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG EBOOK",
            "*** END OF THIS PROJECT GUTENBERG EBOOK",
        ]

        start_idx = 0
        for marker in start_markers:
            idx = text.find(marker)
            if idx >= 0:
                start_idx = idx + len(marker)
                break

        end_idx = len(text)
        for marker in end_markers:
            idx = text.find(marker)
            if idx >= 0:
                end_idx = min(end_idx, idx)

        return text[start_idx:end_idx].strip()

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [self._normalize_text(sentence) for sentence in sentences if len(sentence.strip()) >= self.min_len]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _split_paragraphs(self, text: str) -> List[str]:
        return [
            self._normalize_text(paragraph)
            for paragraph in text.split("\n\n")
            if len(paragraph.strip()) >= self.min_len
        ]

    @staticmethod
    def _stop_requested(stop_event: Optional[object]) -> bool:
        if stop_event is None:
            return False
        if hasattr(stop_event, "is_set"):
            try:
                return bool(stop_event.is_set())
            except Exception:
                return False
        return False

    def _split_gutenberg_chapters(self, text: str) -> List[str]:
        chapter_pattern = re.compile(r"(?im)^\s*chapter\s+([ivxlcdm0-9]+)\b.*$")
        matches = list(chapter_pattern.finditer(text))
        if not matches:
            cleaned = text.strip()
            return [cleaned] if len(cleaned) >= self.min_len else []

        chapters: List[str] = []
        if matches[0].start() > 0:
            prefix = text[: matches[0].start()].strip()
            if len(prefix) >= self.min_len:
                chapters.append(prefix)

        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            chapter = text[start:end].strip()
            if len(chapter) >= self.min_len:
                chapters.append(chapter)

        return chapters

    def _score_chunks(self, chunks: Sequence[str]) -> List[Tuple[float, str]]:
        scored: List[Tuple[float, str]] = []
        for chunk in chunks:
            score = self.score_text(chunk)
            scored.append((score, chunk))
        return scored

    @staticmethod
    def _emit_progress(progress_callback, event: dict) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(event)
        except Exception:
            return

    def score_sentence(self, sentence: str) -> float:
        tokens = self.reader._clean_words(sentence)
        if len(tokens) < self.context_len + 1:
            return 0.0

        total_entropy = 0.0
        n_predictions = 0
        for idx in range(self.context_len, len(tokens)):
            context = tokens[idx - self.context_len : idx]
            distribution = self.contextual_agent.predict_next_distribution(context)
            if not distribution:
                continue

            scores = np.array([max(weight, 1e-8) for _, weight in distribution], dtype=float)
            probs = scores / (scores.sum() + 1e-9)
            entropy = -float(np.sum(probs * np.log(probs + 1e-8)))
            total_entropy += entropy
            n_predictions += 1

        if n_predictions == 0:
            return 0.0
        return total_entropy / n_predictions

    def score_text(self, text: str) -> float:
        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0

        sentence_scores = [self.score_sentence(sentence) for sentence in sentences]
        nonzero_scores = [score for score in sentence_scores if score > 0.0]
        if not nonzero_scores:
            return 0.0
        return float(sum(nonzero_scores) / len(nonzero_scores))

    def _append_examples(self, texts: Sequence[str], separator: str = "\n\n") -> int:
        if not texts:
            return 0

        with open(self.corpus_path, "a", encoding="utf-8") as handle:
            for text in texts:
                handle.write(self._normalize_text(text) + separator)
        return len(texts)

    @staticmethod
    def _select_examples(
        scored: Sequence[Tuple[float, str]],
        top_k: int,
        min_score: float,
    ) -> List[str]:
        ordered = sorted(scored, key=lambda item: item[0], reverse=True)
        above_threshold = [text for score, text in ordered if score >= min_score]
        if above_threshold:
            return above_threshold[:top_k]
        return [text for _, text in ordered[:top_k]]

    def _fetch_text_url(self, url: str, timeout: int = 20) -> str:
        headers = {"User-Agent": "HPM-DatasetTrainingAgent/1.0"}
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        return response.text

    def add_from_text_urls(
        self,
        urls: Sequence[str],
        top_k: int = 100,
        min_score: float = 0.5,
        retrain_epochs: int = 1,
        split_on_paragraphs: bool = True,
    ) -> int:
        scored: List[Tuple[float, str]] = []
        for url in urls:
            print(f"Fetching text corpus: {url}")
            text = self._fetch_text_url(url)
            if "project gutenberg" in url.lower():
                text = self._strip_gutenberg_boilerplate(text)

            chunks = self._split_paragraphs(text) if split_on_paragraphs else self._split_sentences(text)

            print(f"  Scoring {len(chunks)} chunks...")
            for idx, chunk in enumerate(chunks):
                score = self.score_text(chunk)
                scored.append((score, chunk))
                if idx > 0 and idx % 100 == 0:
                    print(f"    Scored {idx} chunks")

        selected = self._select_examples(scored, top_k=top_k, min_score=min_score)
        if not selected:
            print("No informative examples found.")
            return 0

        added = self._append_examples(selected)
        print(f"Added {added} text-url examples to {self.corpus_path}")
        self.reader.retrain_on_new_data(epochs=retrain_epochs)
        return added

    def train_gutenberg_cycle(
        self,
        book_ids: Optional[Sequence[int]] = None,
        top_k_per_chapter: int = 8,
        min_score: float = 0.05,
        retrain_epochs: int = 1,
        split_on_paragraphs: bool = False,
        stop_event: Optional[object] = None,
        repeat_books: bool = True,
        report_progress: bool = True,
        max_books: Optional[int] = None,
        progress_callback=None,
        maintenance_callback=None,
    ) -> List[dict]:
        """
        Stream Gutenberg books chapter by chapter.

        If repeat_books is True, the provided book list is cycled until stop_event
        is set or max_books is reached. Otherwise the list is processed once.
        """
        ids = [int(book_id) for book_id in (book_ids or self.curated_gutenberg_book_ids())]
        if not ids:
            return []

        reports: List[dict] = []
        books_processed = 0
        while True:
            for book_id in ids:
                if self._stop_requested(stop_event):
                    return reports
                if max_books is not None and books_processed >= max_books:
                    return reports

                url = self._gutenberg_url(book_id)
                self._emit_progress(progress_callback, {
                    "type": "book_start",
                    "book_id": book_id,
                    "url": url,
                    "books_processed": books_processed,
                })
                if report_progress:
                    print(f"Fetching Gutenberg book {book_id}: {url}")

                text = self._fetch_text_url(url)
                text = self._strip_gutenberg_boilerplate(text)
                chapters = self._split_gutenberg_chapters(text)
                if not chapters:
                    chapters = [text.strip()]

                book_added = 0
                chapter_reports: List[dict] = []
                selected_texts: List[str] = []
                for chapter_idx, chapter_text in enumerate(chapters, start=1):
                    if self._stop_requested(stop_event):
                        break

                    self._emit_progress(progress_callback, {
                        "type": "chapter_start",
                        "book_id": book_id,
                        "chapter": chapter_idx,
                        "chapters": len(chapters),
                    })
                    chunks = self._split_paragraphs(chapter_text) if split_on_paragraphs else self._split_sentences(chapter_text)
                    if not chunks:
                        self._emit_progress(progress_callback, {
                            "type": "chapter_skip",
                            "book_id": book_id,
                            "chapter": chapter_idx,
                            "reason": "no_chunks",
                        })
                        continue

                    if report_progress:
                        print(f"  Chapter {chapter_idx}: scoring {len(chunks)} chunks")

                    scored = self._score_chunks(chunks)
                    selected = self._select_examples(scored, top_k=top_k_per_chapter, min_score=min_score)
                    if not selected:
                        self._emit_progress(progress_callback, {
                            "type": "chapter_done",
                            "book_id": book_id,
                            "chapter": chapter_idx,
                            "chunks": len(chunks),
                            "selected": 0,
                            "added": 0,
                        })
                        if report_progress:
                            print(f"  Chapter {chapter_idx}: no informative chunks found")
                        chapter_reports.append(
                            {
                                "book_id": book_id,
                                "chapter": chapter_idx,
                                "chunks": len(chunks),
                                "selected": 0,
                                "added": 0,
                            }
                        )
                        continue

                    added = self._append_examples(selected)
                    book_added += added
                    selected_texts.extend(selected)
                    chapter_reports.append(
                        {
                            "book_id": book_id,
                            "chapter": chapter_idx,
                            "chunks": len(chunks),
                            "selected": len(selected),
                            "added": added,
                        }
                    )

                    if report_progress:
                        print(f"  Chapter {chapter_idx}: added {added} chunks")
                    self._emit_progress(progress_callback, {
                        "type": "chapter_done",
                        "book_id": book_id,
                        "chapter": chapter_idx,
                        "chunks": len(chunks),
                        "selected": len(selected),
                        "added": added,
                    })

                    if added > 0:
                        self.reader.retrain_on_new_data(epochs=retrain_epochs)

                report = {
                    "book_id": book_id,
                    "chapters": len(chapters),
                    "added": book_added,
                    "chapter_reports": chapter_reports,
                }
                if maintenance_callback is not None and selected_texts and hasattr(self.reader, "maintenance_cycle"):
                    try:
                        maintenance_report = self.reader.maintenance_cycle(
                            sentences=selected_texts[-top_k_per_chapter:],
                            hydrate_limit=None,
                            retrain_epochs=retrain_epochs,
                            enable_causal=False,
                        )
                    except Exception as exc:  # pragma: no cover - defensive maintenance path
                        maintenance_report = {"error": str(exc)}
                    report["maintenance_report"] = maintenance_report
                    try:
                        maintenance_callback({
                            "type": "maintenance_done",
                            "book_id": book_id,
                            "report": maintenance_report,
                        })
                    except Exception:  # pragma: no cover - defensive maintenance path
                        pass
                reports.append(report)
                books_processed += 1

                if report_progress:
                    print(
                        f"Finished book {book_id}: chapters={len(chapters)}, added={book_added}, retrain_epochs={retrain_epochs}"
                    )
                self._emit_progress(progress_callback, {
                    "type": "book_done",
                    "book_id": book_id,
                    "chapters": len(chapters),
                    "added": book_added,
                    "books_processed": books_processed,
                })

            if not repeat_books or self._stop_requested(stop_event):
                break
            if max_books is not None and books_processed >= max_books:
                break

        self._emit_progress(progress_callback, {
            "type": "cycle_done",
            "books_processed": books_processed,
            "repeat_books": repeat_books,
        })
        return reports

    def add_from_gutenberg(
        self,
        book_ids: Optional[Sequence[int]] = None,
        top_k: int = 100,
        min_score: float = 0.5,
        retrain_epochs: int = 1,
        split_on_paragraphs: bool = False,
    ) -> int:
        if not book_ids:
            book_ids = self.curated_gutenberg_book_ids()
        urls = [self._gutenberg_url(book_id) for book_id in book_ids]
        return self.add_from_text_urls(
            urls,
            top_k=top_k,
            min_score=min_score,
            retrain_epochs=retrain_epochs,
            split_on_paragraphs=split_on_paragraphs,
        )

    def add_from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        max_examples: int = 1000,
        top_k: int = 100,
        text_column: str = "text",
        min_score: float = 0.5,
        dataset_config: Optional[str] = None,
        retrain_epochs: int = 1,
    ) -> int:
        if load_dataset is None:
            raise ImportError("datasets is not installed. Run: pip install datasets")

        print(f"Loading dataset '{dataset_name}' split '{split}'...")
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        if len(dataset) > max_examples:
            dataset = dataset.shuffle(seed=42).select(range(max_examples))

        print(f"Scoring {len(dataset)} examples...")
        scored: List[Tuple[float, str]] = []
        for idx, example in enumerate(dataset):
            text = str(example.get(text_column, "")).strip()
            if len(text) < self.min_len:
                continue
            score = self.score_text(text)
            scored.append((score, text))
            if idx > 0 and idx % 100 == 0:
                print(f"  Scored {idx} examples")

        selected = self._select_examples(scored, top_k=top_k, min_score=min_score)
        if not selected:
            print("No informative examples found.")
            return 0

        added = self._append_examples(selected)
        print(f"Added {added} dataset examples to {self.corpus_path}")
        self.reader.retrain_on_new_data(epochs=retrain_epochs)
        return added

    def add_from_local_file(
        self,
        file_path: str,
        top_k: int = 100,
        min_score: float = 0.5,
        retrain_epochs: int = 1,
        split_on_paragraphs: bool = True,
    ) -> int:
        text = Path(file_path).read_text(encoding="utf-8")
        chunks = self._split_paragraphs(text) if split_on_paragraphs else self._split_sentences(text)
        print(f"Scoring {len(chunks)} chunks from {file_path}...")

        scored: List[Tuple[float, str]] = []
        for chunk in chunks:
            score = self.score_text(chunk)
            scored.append((score, chunk))

        selected = self._select_examples(scored, top_k=top_k, min_score=min_score)
        if not selected:
            print("No informative examples found.")
            return 0

        added = self._append_examples(selected)
        print(f"Added {added} chunks from {file_path} to {self.corpus_path}")
        self.reader.retrain_on_new_data(epochs=retrain_epochs)
        return added
