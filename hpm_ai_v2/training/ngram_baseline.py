"""Smoothed n-gram baseline using the language-pilot tokenizer."""
from __future__ import annotations

import math
from collections import OrderedDict
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np

from hpm_ai_v2.training.language_prediction import BOUNDARY, UNK, split_sentences, tokenize


class NGramBaseline:
    def __init__(self, order: int = 2, alpha: float = .1, smoothing: str = "add_alpha",
                 min_word_count: int = 1):
        if order < 1 or alpha <= 0 or min_word_count < 1:
            raise ValueError("order must be positive and alpha must be > 0")
        if smoothing not in {"add_alpha", "interpolated"}:
            raise ValueError("smoothing must be add_alpha or interpolated")
        self.order, self.alpha, self.min_word_count = order, alpha, min_word_count
        self.smoothing = smoothing
        self.counts = defaultdict(Counter)
        self.vocab = set()
        self._vector_cache = OrderedDict()

    def fit(self, documents: Iterable[str]):
        documents = [[tokenize(sentence) for sentence in split_sentences(text)] for text in documents]
        self.word_counts = Counter(word for sentences in documents for words in sentences for word in words)
        for sentences in documents:
            for raw_words in sentences:
                words = [word if self.word_counts[word] >= self.min_word_count else UNK
                         for word in raw_words] + [BOUNDARY]
                self.vocab.update(words)
                padded = [BOUNDARY] * (self.order - 1) + words
                for i in range(self.order - 1, len(padded)):
                    for context_order in range(self.order):
                        context = tuple(padded[i-context_order:i]) if context_order else ()
                        self.counts[context][padded[i]] += 1
        self.vocab.add(UNK)
        self.vocab = sorted(self.vocab)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        return self

    def predict_proba(self, context):
        mapped = tuple(word if self.word_counts.get(word, 0) >= self.min_word_count else UNK
                       for word in context)
        key = mapped[-(self.order - 1):] if self.order > 1 else ()
        while key not in self.counts and key:
            key = key[1:]
        if key in self._vector_cache:
            self._vector_cache.move_to_end(key)
            return self._vector_cache[key]
        if self.smoothing == "interpolated":
            probabilities = self._interpolated_distribution(key)
        else:
            counts = self.counts.get(key, Counter())
            total = sum(counts.values())
            probabilities = np.full(len(self.vocab), self.alpha / (total + self.alpha * len(self.vocab)))
            if total:
                for word, value in counts.items():
                    probabilities[self.index[word]] += value / (total + self.alpha * len(self.vocab))
        self._vector_cache[key] = probabilities
        if len(self._vector_cache) > 2048:
            self._vector_cache.popitem(last=False)
        return probabilities

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        if hasattr(self, "_vector_cache"):
            self._vector_cache.clear()

    def _interpolated_distribution(self, context):
        if not context:
            counts = self.counts.get((), Counter())
            total = sum(counts.values())
            probabilities = np.full(len(self.vocab), self.alpha / (total + self.alpha * len(self.vocab)))
            for word, value in counts.items():
                probabilities[self.index[word]] += value / (total + self.alpha * len(self.vocab))
            return probabilities
        counts = self.counts.get(context, Counter())
        total = sum(counts.values())
        if not total:
            return self._interpolated_distribution(context[1:])
        weight = total / (total + self.alpha)
        probabilities = (1.0 - weight) * self._interpolated_distribution(context[1:])
        for word, value in counts.items():
            probabilities[self.index[word]] += weight * value / total
        return probabilities

    def _interpolated_probability(self, context, target):
        if not context:
            counts = self.counts.get((), Counter())
            return (counts.get(target, 0) + self.alpha) / (sum(counts.values()) + self.alpha * len(self.vocab))
        counts = self.counts.get(context, Counter())
        total = sum(counts.values())
        if not total:
            return self._interpolated_probability(context[1:], target)
        weight = total / (total + self.alpha)
        return weight * counts.get(target, 0) / total + (1-weight) * self._interpolated_probability(context[1:], target)

    def score_cross_entropy(self, documents: Iterable[str]) -> float:
        """Score log loss without computing top-k rankings for every token."""
        nll = count = 0
        for text in documents:
            for sentence in split_sentences(text):
                raw_seq = tokenize(sentence)
                seq = [word if self.word_counts.get(word, 0) >= self.min_word_count else UNK
                       for word in raw_seq] + [BOUNDARY]
                for i, raw_target in enumerate(seq):
                    target = raw_target if raw_target in self.index else UNK
                    context = [w if w in self.index else UNK
                               for w in seq[max(0, i-self.order+1):i]]
                    if self.smoothing == "interpolated":
                        probability = self._interpolated_probability(tuple(context), target)
                    else:
                        counts = self.counts.get(tuple(context), Counter())
                        probability = (counts.get(target, 0) + self.alpha) / (
                            sum(counts.values()) + self.alpha * len(self.vocab))
                    nll -= math.log(max(float(probability), 1e-12))
                    count += 1
        return nll / max(count, 1)

    def evaluate(self, documents: Iterable[str]):
        nll = correct = top5 = count = 0
        surface_correct = surface_top5 = known_correct = known_top5 = 0
        lexical_tokens = oov_tokens = known_tokens = 0
        for text in documents:
            for sentence in split_sentences(text):
                surface_sequence = tokenize(sentence) + [BOUNDARY]
                seq = [word if word == BOUNDARY or self.word_counts.get(word, 0) >= self.min_word_count else UNK
                       for word in surface_sequence]
                for i, surface_target in enumerate(surface_sequence):
                    target = seq[i] if seq[i] in self.index else UNK
                    context = [w if w in self.index else UNK for w in seq[max(0, i-self.order+1):i]]
                    idx = self.index[target]
                    p = self.predict_proba(context)
                    nll -= math.log(max(float(p[idx]), 1e-12))
                    correct += int(int(np.argmax(p)) == idx)
                    ranked_indices = np.argsort(p)[-5:]
                    top5 += int(idx in ranked_indices)
                    predicted_word = self.vocab[int(np.argmax(p))]
                    ranked_words = {self.vocab[int(index)] for index in ranked_indices}
                    surface_correct += int(predicted_word == surface_target)
                    surface_top5 += int(surface_target in ranked_words)
                    if surface_target != BOUNDARY:
                        lexical_tokens += 1
                        if surface_target in self.index and surface_target != UNK:
                            known_tokens += 1
                            known_correct += int(predicted_word == surface_target)
                            known_top5 += int(surface_target in ranked_words)
                        else:
                            oov_tokens += 1
                    count += 1
        mean = nll / max(count, 1)
        return {"tokens": count, "cross_entropy_nats": mean, "perplexity": math.exp(mean),
                "target_space": "vocabulary_mapped",
                "top1_accuracy": correct / max(count, 1), "top5_accuracy": top5 / max(count, 1),
                "surface_top1_accuracy": surface_correct / max(count, 1),
                "surface_top5_accuracy": surface_top5 / max(count, 1),
                "known_word_top1_accuracy": known_correct / max(known_tokens, 1),
                "known_word_top5_accuracy": known_top5 / max(known_tokens, 1),
                "known_word_tokens": known_tokens, "oov_word_tokens": oov_tokens,
                "oov_word_rate": oov_tokens / max(lexical_tokens, 1)}
