"""Interpolated absolute-discount Kneser-Ney n-gram baseline.

The model shares the language pilot's sentence splitter, tokenizer, unknown
word mapping and end-of-sentence target so its cross-entropy is comparable.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

from hpm_ai_v2.training.language_prediction import BOUNDARY, UNK, split_sentences, tokenize


class KneserNeyBaseline:
    def __init__(self, order: int = 3, discount: float = 0.75, min_word_count: int = 2,
                 unknown_continuation_prior: float = 0.0):
        if order not in (2, 3):
            raise ValueError("Kneser-Ney baseline supports order 2 or 3")
        if not 0.0 < discount < 1.0 or min_word_count < 1:
            raise ValueError("discount must be in (0, 1) and min_word_count positive")
        if unknown_continuation_prior < 0:
            raise ValueError("unknown_continuation_prior must be non-negative")
        self.order = order
        self.discount = float(discount)
        self.min_word_count = min_word_count
        self.unknown_continuation_prior = float(unknown_continuation_prior)
        self.word_counts: Counter[str] = Counter()
        self.vocab: set[str] = set()
        self.context_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {}
        self.context_totals: dict[int, dict[tuple[str, ...], int]] = {}
        self.context_types: dict[int, dict[tuple[str, ...], int]] = {}
        self.continuation_counts: Counter[str] = Counter()
        self.continuation_total = 0

    def fit(self, documents: Iterable[str]) -> "KneserNeyBaseline":
        texts = list(documents)
        raw_sentences = [tokenize(sentence) for text in texts for sentence in split_sentences(text)]
        self.word_counts = Counter(token for sentence in raw_sentences for token in sentence)
        self.vocab = {BOUNDARY, UNK} | {
            token for token, count in self.word_counts.items() if count >= self.min_word_count
        }
        counts = {n: defaultdict(Counter) for n in range(1, self.order + 1)}
        bigram_predecessors: dict[str, set[str]] = defaultdict(set)
        for raw in raw_sentences:
            sequence = [self._map(token) for token in raw] + [BOUNDARY]
            padded = [BOUNDARY] * (self.order - 1) + sequence
            for i in range(self.order - 1, len(padded)):
                target = padded[i]
                for n in range(1, self.order + 1):
                    context = tuple(padded[i - n + 1:i]) if n > 1 else ()
                    counts[n][context][target] += 1
                if i > 0:
                    bigram_predecessors[target].add(padded[i - 1])
        self.context_counts = {n: dict(rows) for n, rows in counts.items()}
        self.context_totals = {
            n: {context: sum(targets.values()) for context, targets in rows.items()}
            for n, rows in self.context_counts.items()
        }
        self.context_types = {
            n: {context: len(targets) for context, targets in rows.items()}
            for n, rows in self.context_counts.items()
        }
        # Lower-order unigram probabilities are continuation probabilities,
        # not raw token frequencies, as required by Kneser-Ney.
        self.continuation_counts = Counter({
            word: len(predecessors) for word, predecessors in bigram_predecessors.items()
        })
        self.continuation_total = sum(self.continuation_counts.values())
        return self

    def _map(self, word: str) -> str:
        if word in (BOUNDARY, UNK):
            return word
        return word if self.word_counts.get(word, 0) >= self.min_word_count else UNK

    def probability(self, context: Iterable[str], target: str) -> float:
        if not self.vocab:
            raise RuntimeError("fit() must be called before probability()")
        mapped_context = tuple(self._map(word) for word in context)[-(self.order - 1):]
        target = self._map(target)
        return self._probability(mapped_context, target, self.order)

    def _probability(self, context: tuple[str, ...], target: str, n: int) -> float:
        if n <= 1:
            prior = self.unknown_continuation_prior if target == UNK else 0.0
            return (self.continuation_counts.get(target, 0) + prior) / max(
                self.continuation_total + self.unknown_continuation_prior, 1.0
            )
        context = context[-(n - 1):]
        rows = self.context_counts[n].get(context)
        if not rows:
            return self._probability(context[1:], target, n - 1)
        total = self.context_totals[n][context]
        discounted = max(rows.get(target, 0) - self.discount, 0.0) / total
        backoff = self.discount * self.context_types[n][context] / total
        return discounted + backoff * self._probability(context[1:], target, n - 1)

    def score_cross_entropy(self, documents: Iterable[str]) -> dict:
        nll = tokens = 0
        for text in documents:
            for sentence in split_sentences(text):
                sequence = [self._map(word) for word in tokenize(sentence)] + [BOUNDARY]
                padded = [BOUNDARY] * (self.order - 1) + sequence
                for index in range(self.order - 1, len(padded)):
                    context = padded[index - self.order + 1:index]
                    probability = self.probability(context, padded[index])
                    nll -= math.log(max(probability, 1e-12))
                    tokens += 1
        mean = nll / max(tokens, 1)
        return {"tokens": tokens, "cross_entropy_nats": mean, "perplexity": math.exp(mean),
                "order": self.order, "discount": self.discount,
                "unknown_continuation_prior": self.unknown_continuation_prior,
                "min_word_count": self.min_word_count}
