#!/usr/bin/env python3
"""
Web Agent for the HPM multi-agent reader.
Fetches text from URLs or RSS feeds, scores sentences with the contextual model,
appends informative sentences to the corpus, and triggers incremental retraining.
"""

from __future__ import annotations

import re
from typing import List, Sequence

import feedparser
import numpy as np
import requests
from bs4 import BeautifulSoup


class WebAgent:
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

    def fetch_text_from_url(self, url: str) -> str:
        try:
            headers = {"User-Agent": "HPM-WebAgent/1.0"}
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return " ".join(chunk for chunk in chunks if chunk)
        except Exception as exc:
            print(f"Error fetching {url}: {exc}")
            return ""

    def fetch_from_rss(self, feed_url: str, max_entries: int = 10) -> List[str]:
        urls: List[str] = []
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_entries]:
                if "link" in entry:
                    urls.append(entry.link)
        except Exception as exc:
            print(f"Error reading RSS feed {feed_url}: {exc}")
        return urls

    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [sentence.strip() for sentence in sentences if len(sentence.strip()) >= self.min_len]

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

    def add_informative_sentences(
        self,
        urls: Sequence[str],
        top_k: int = 100,
        min_score: float = 0.5,
        retrain_epochs: int = 1,
    ) -> int:
        all_scored = []
        for url in urls:
            print(f"Processing: {url}")
            text = self.fetch_text_from_url(url)
            if not text:
                continue
            for sentence in self.split_sentences(text):
                score = self.score_sentence(sentence)
                if score >= min_score:
                    all_scored.append((score, sentence))

        all_scored.sort(key=lambda item: item[0], reverse=True)
        new_sentences = [sentence for _, sentence in all_scored[:top_k]]
        if not new_sentences:
            print("No informative sentences found.")
            return 0

        with open(self.corpus_path, "a", encoding="utf-8") as handle:
            for sentence in new_sentences:
                handle.write(sentence + "\n")

        print(f"Added {len(new_sentences)} informative sentences to {self.corpus_path}")
        self.reader.retrain_on_new_data(epochs=retrain_epochs)
        return len(new_sentences)

    def process_rss_feed(
        self,
        feed_url: str,
        top_k: int = 100,
        max_entries: int = 10,
        min_score: float = 0.5,
        retrain_epochs: int = 1,
    ) -> int:
        urls = self.fetch_from_rss(feed_url, max_entries=max_entries)
        if not urls:
            print("No URLs found in RSS feed.")
            return 0
        return self.add_informative_sentences(
            urls,
            top_k=top_k,
            min_score=min_score,
            retrain_epochs=retrain_epochs,
        )
