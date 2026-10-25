"""Scoring, evaluation, and Observer calibration methods for the language model."""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np

from hpm_ai_v2.training.language_constants import BOUNDARY, TOKEN_RE, UNK
from hpm_ai_v2.training.language_corpus import split_sentences, tokenize


class LanguageEvaluationMixin:
    def evaluate_characters(self, documents: Iterable[str]) -> dict:
        """Held-out next-character score from the same encoder-free learner."""
        vocab_size = len(self.char_vocab)
        nll = correct = count = 0
        top_character_by_context: dict[tuple[str, ...], str | None] = {}
        for text in documents:
            for sentence in split_sentences(text):
                seq = [BOUNDARY] * (self.max_order - 1) + list(sentence.lower()) + [BOUNDARY]
                for i in range(self.max_order - 1, len(seq)):
                    raw_target = seq[i]
                    target = raw_target if raw_target in self.char_vocab else UNK
                    context = tuple(seq[i-self.max_order+1:i]) if self.max_order > 1 else ()
                    context = tuple(ch if ch in self.char_vocab else UNK for ch in context)
                    p = self._char_probability(context, target, vocab_size)
                    nll -= math.log(max(p, 1e-12))
                    if context not in top_character_by_context:
                        candidates = set()
                        probe = context
                        while True:
                            candidates.update(ch for ch, _ in self.char_counts.get(probe, Counter()).most_common(1))
                            if not probe:
                                break
                            probe = probe[1:]
                        ranked = sorted(
                            candidates,
                            key=lambda ch: self._char_probability(context, ch, vocab_size),
                            reverse=True,
                        )
                        top_character_by_context[context] = ranked[0] if ranked else None
                    correct += int(top_character_by_context[context] == target)
                    count += 1
        mean = nll / max(count, 1)
        return {"characters": count, "cross_entropy_nats": mean, "perplexity": math.exp(mean),
                "top1_accuracy": correct / max(count, 1)}

    def _char_probability(self, context: tuple[str, ...], target: str, vocab_size: int) -> float:
        return self._weighted_probability(context, target, character=True)

    def score_cross_entropy(self, documents: Iterable[str], *, character: bool = False,
                             max_events: int | None = None) -> float:
        """Score next-unit log loss, optionally using a deterministic strided sample."""
        documents = list(documents)
        total_events = sum(
            (len(sentence) if character else len(tokenize(sentence))) + 1
            for text in documents for sentence in split_sentences(text)
        )
        stride = max(1, math.ceil(total_events / max_events)) if max_events else 1
        nll = count = 0
        event_index = 0
        for text in documents:
            for sentence in split_sentences(text):
                if character:
                    sequence = [BOUNDARY] * (self.max_order - 1) + list(sentence.lower()) + [BOUNDARY]
                    for i in range(self.max_order - 1, len(sequence)):
                        if event_index % stride:
                            event_index += 1
                            continue
                        event_index += 1
                        target = sequence[i] if sequence[i] in self.char_vocab else UNK
                        context = tuple(sequence[i-self.max_order+1:i]) if self.max_order > 1 else ()
                        context = tuple(ch if ch in self.char_vocab else UNK for ch in context)
                        probability = self._char_probability(context, target, len(self.char_vocab))
                        nll -= math.log(max(probability, 1e-12))
                        count += 1
                else:
                    sequence = [self._map(word) for word in tokenize(sentence)] + [BOUNDARY]
                    for i, target in enumerate(sequence):
                        if event_index % stride:
                            event_index += 1
                            continue
                        event_index += 1
                        context = tuple(sequence[max(0, i-self.max_order+1):i])
                        probability = self._probability(context, target)
                        nll -= math.log(max(probability, 1e-12))
                        count += 1
        return nll / max(count, 1)

    def score_surface_cross_entropy(
        self, documents: Iterable[str], *, max_events: int | None = None,
    ) -> float:
        """Score exact next-word spellings, including OOV forms and EOS."""
        documents = list(documents)
        total_events = sum(
            len(tokenize(sentence)) + 1
            for text in documents for sentence in split_sentences(text)
        )
        stride = max(1, math.ceil(total_events / max_events)) if max_events else 1
        nll = count = event_index = 0
        for text in documents:
            for sentence in split_sentences(text):
                sequence = tokenize(sentence) + [BOUNDARY]
                for index, target in enumerate(sequence):
                    if event_index % stride:
                        event_index += 1
                        continue
                    event_index += 1
                    context = sequence[max(0, index - self.max_order + 1):index]
                    probability = self.surface_word_probability(context, target)
                    nll -= math.log(max(probability, 1e-300))
                    count += 1
        return nll / max(count, 1)

    def tune_observer_weights(self, validation_documents: Iterable[str], *, character: bool = False,
                              passes: int = 2, max_events: int = 20_000,
                              progress_callback: Callable[[dict], None] | None = None) -> dict:
        """Tune Observer level weights by validation next-unit log loss only."""
        documents = list(validation_documents)
        registry = self.char_level_nodes if character else self.word_level_nodes
        if not registry or self.observer is None:
            raise RuntimeError("pattern Observer is not initialized")
        initial = {order: self.observer.get_weight(node.id) for order, node in registry.items()}
        total_events = sum(
            (len(sentence) if character else len(tokenize(sentence))) + 1
            for text in documents for sentence in split_sentences(text)
        )
        before = self.score_cross_entropy(documents, character=character)
        candidates = (0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)
        completed = 0
        for _ in range(max(1, passes)):
            changed = False
            for order, node in registry.items():
                current = self.observer.get_weight(node.id)
                best_weight, best_loss = current, self.score_cross_entropy(
                    documents, character=character, max_events=max_events)
                for candidate in (*candidates, current):
                    self.observer.set_weight(node.id, candidate)
                    loss = self.score_cross_entropy(documents, character=character,
                                                    max_events=max_events)
                    if loss < best_loss - 1e-12:
                        best_weight, best_loss = candidate, loss
                    completed += 1
                    if progress_callback:
                        progress_callback({"unit": "character" if character else "word",
                                           "order": order, "candidate_weight": candidate,
                                           "sample_validation_cross_entropy": loss,
                                           "completed_candidates": completed})
                self.observer.set_weight(node.id, best_weight)
                changed |= best_weight != current
            if not changed:
                break
        final = {order: self.observer.get_weight(node.id) for order, node in registry.items()}
        after = self.score_cross_entropy(documents, character=character)
        rolled_back = after > before + 1e-12
        if rolled_back:
            for order, node in registry.items():
                self.observer.set_weight(node.id, initial[order])
            final = dict(initial)
            after = self.score_cross_entropy(documents, character=character)
        return {"unit": "character" if character else "word", "validation_cross_entropy_before": before,
                "validation_cross_entropy_after": after,
                "initial_weights": {str(order): weight for order, weight in initial.items()},
                "selected_weights": {str(order): weight for order, weight in final.items()},
                "validation_events": total_events,
                "calibration_events": min(max_events, total_events),
                "rolled_back_on_full_validation": rolled_back,
                "passes": max(1, passes)}

    def _distribution_for_context(self, context: tuple[str, ...]) -> dict[str, float]:
        """Sparse additive-smoothed interpolation with suffix backoff."""
        if not context:
            counts = self.counts.get((), Counter())
            denom = sum(counts.values()) + self.alpha * len(self.vocab)
            return {word: (counts.get(word, 0) + self.alpha) / denom for word in self.vocab}
        base = self._distribution_for_context(context[1:])
        counts = self.counts.get(context, Counter())
        total = sum(counts.values())
        local_mass = total / (total + self.alpha)
        if not total:
            plain = base
        else:
            plain = {word: local_mass * counts.get(word, 0) / total + (1.0 - local_mass) * base[word]
                     for word in self.vocab}
        mixed = self._class_mixed_distribution(context, plain_distribution=plain)
        return plain if mixed is None else mixed

    def evaluate(self, documents: Iterable[str], *, include_surface: bool = True) -> dict:
        nll, surface_nll, correct, top5, count = 0.0, 0.0, 0, 0, 0
        surface_correct = surface_top5 = known_correct = known_top5 = 0
        lexical_tokens = oov_tokens = known_tokens = 0
        uncertainty_values, token_surprisal = [], []
        for text in documents:
            for sentence in split_sentences(text):
                surface_sequence = tokenize(sentence) + [BOUNDARY]
                seq = [self._map(word) for word in surface_sequence]
                for i, target in enumerate(seq):
                    context = seq[max(0, i - self.max_order + 1):i]
                    mapped = tuple(self._map(w.lower()) for w in context)[-(self.max_order - 1):] if self.max_order > 1 else ()
                    p_target = self._probability(mapped, target)
                    if include_surface:
                        surface_probability = self.surface_word_probability(
                            surface_sequence[max(0, i - self.max_order + 1):i],
                            surface_sequence[i],
                        )
                    context_node = self.context_nodes.get(mapped)
                    uncertainty_values.append(float(context_node.sigma[0]) if context_node is not None else 0.0)
                    token_surprisal.append(-math.log(max(p_target, 1e-12)))
                    candidates = self._top_candidates(mapped, 5)
                    candidate_words = sorted(candidates, key=self.word_to_index.__getitem__)
                    candidate_probabilities = np.asarray(
                        [self._probability(mapped, word) for word in candidate_words]
                    )
                    candidate_order = np.argsort(candidate_probabilities)
                    top1_word = candidate_words[int(np.argmax(candidate_probabilities))] if candidate_words else None
                    ranked = [candidate_words[int(index)] for index in candidate_order[-5:]]
                    nll -= math.log(max(p_target, 1e-12))
                    if include_surface:
                        surface_nll -= math.log(max(surface_probability, 1e-300))
                    correct += int(top1_word == target)
                    top5 += int(target in ranked)
                    surface_target = surface_sequence[i]
                    surface_correct += int(top1_word == surface_target)
                    surface_top5 += int(surface_target in ranked)
                    if surface_target != BOUNDARY:
                        lexical_tokens += 1
                        if surface_target in self.word_to_index and surface_target != UNK:
                            known_tokens += 1
                            known_correct += int(top1_word == surface_target)
                            known_top5 += int(surface_target in ranked)
                        else:
                            oov_tokens += 1
                    count += 1
        sigma_correlation = self._spearman(uncertainty_values, token_surprisal)
        calibration = []
        ordered = sorted(zip(uncertainty_values, token_surprisal), key=lambda pair: pair[0])
        for i in range(10):
            segment = ordered[i * len(ordered) // 10:(i + 1) * len(ordered) // 10]
            if segment:
                calibration.append({"bin": i + 1, "tokens": len(segment),
                                    "mean_sigma": sum(x for x, _ in segment) / len(segment),
                                    "mean_surprisal_nats": sum(y for _, y in segment) / len(segment)})
        metrics = {"tokens": count, "cross_entropy_nats": nll / max(count, 1), "perplexity": math.exp(nll / max(count, 1)),
                "target_space": "vocabulary_mapped",
                "top1_accuracy": correct / max(count, 1), "top5_accuracy": top5 / max(count, 1),
                "surface_top1_accuracy": surface_correct / max(count, 1),
                "surface_top5_accuracy": surface_top5 / max(count, 1),
                "known_word_top1_accuracy": known_correct / max(known_tokens, 1),
                "known_word_top5_accuracy": known_top5 / max(known_tokens, 1),
                "known_word_tokens": known_tokens, "oov_word_tokens": oov_tokens,
                "oov_word_rate": oov_tokens / max(lexical_tokens, 1),
                "sigma_surprisal_spearman": sigma_correlation, "sigma_calibration_deciles": calibration}
        if include_surface:
            surface_cross_entropy = surface_nll / max(count, 1)
            metrics.update({"surface_cross_entropy_nats": surface_cross_entropy,
                            "surface_perplexity": math.exp(min(surface_cross_entropy, 700)),
                            "surface_target_space": "exact_surface_word_and_eos"})
        return metrics

    @staticmethod
    def _spearman(left: list[float], right: list[float]) -> float | None:
        if len(left) < 2:
            return None

        def ranks(values):
            order = sorted(range(len(values)), key=values.__getitem__)
            result = [0.0] * len(values)
            i = 0
            while i < len(order):
                j = i + 1
                while j < len(order) and values[order[j]] == values[order[i]]:
                    j += 1
                rank = (i + j - 1) / 2.0
                for k in order[i:j]:
                    result[k] = rank
                i = j
            return np.asarray(result)

        x, y = ranks(left), ranks(right)
        if np.std(x) == 0 or np.std(y) == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])
