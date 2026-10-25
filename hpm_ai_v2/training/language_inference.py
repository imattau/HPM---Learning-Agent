"""Lexical spelling, probability, and next-unit inference methods."""
from __future__ import annotations

import math
import heapq
from collections import Counter, OrderedDict
from typing import Callable, Iterable

import numpy as np

from hpm_ai_v2.training.language_constants import BOUNDARY, TOKEN_RE, UNK
from hpm_ai_v2.training.language_corpus import split_sentences, tokenize

SPELL_BOW = "<bow>"
SPELL_EOW = "</w>"
SPELL_UNK = "<unk-char>"
CHARACTER_WORD_MIX_CANDIDATES = (0.0, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5)


class LanguageInferenceMixin:
    def _weighted_distribution(self, context: tuple[str, ...], *, character: bool = False) -> dict[str, float]:
        vocabulary = self.char_vocab if character else self.vocab
        return {unit: self._weighted_probability(context, unit, character=character)
                for unit in vocabulary}

    def _spelling_signature(self) -> tuple:
        char_weights = tuple(self.observer.get_weight(node.id)
                             for _, node in sorted(self.char_level_nodes.items())) if self.observer else ()
        word_weights = tuple(self.observer.get_weight(node.id)
                             for _, node in sorted(self.word_level_nodes.items())) if self.observer else ()
        # `min_word_count` determines which lexical strings are enumerated by
        # the known-vocabulary branch. Keep this key constant-size: building a
        # tuple of every vocabulary item on every scored token is quadratic
        # overhead on realistic corpora.
        return (self.alpha, self.char_alpha, word_weights, char_weights, self.min_word_count)

    def _ensure_spelling_cache_signature(self) -> None:
        signature = self._spelling_signature()
        if self._spelling_cache_signature != signature:
            self._spelling_cache_signature = signature
            self._spelling_word_probability_cache.clear()
            self._spelling_oov_mass_cache.clear()
            self._spelling_oov_candidate_cache.clear()
            self._spelling_known_candidate_cache.clear()

    def _spelling_char_probability(self, context: tuple[str, ...], target: str) -> float:
        """Probability of a spelling character under the word-started char LM."""
        counts_table = self.spelling_char_counts
        vocabulary = self.spelling_char_vocab
        if not vocabulary:
            return 0.0
        mapped_target = target if target in vocabulary else SPELL_UNK
        mapped_context = tuple(char if char == SPELL_BOW or char in vocabulary else SPELL_UNK
                               for char in context)
        unigram = counts_table.get((), Counter())
        alpha = self.char_alpha
        total = sum(unigram.values())
        probability = (unigram.get(mapped_target, 0) + alpha) / (
            total + alpha * len(vocabulary)
        )
        weights = [self.observer.get_weight(self.char_level_nodes[order].id)
                   for order in range(self.max_order)] if self.observer else [0.5] * self.max_order
        probability = weights[0] * probability + (1.0 - weights[0]) / len(vocabulary)
        for order in range(1, min(len(mapped_context), self.max_order - 1) + 1):
            local = counts_table.get(mapped_context[-order:], Counter())
            local_total = sum(local.values())
            if not local_total:
                continue
            evidence = local_total / (local_total + alpha)
            mass = weights[order] * evidence
            probability = (mass * local.get(mapped_target, 0) / local_total
                           + (1.0 - mass) * probability)
        return probability

    def _spelling_word_probability(self, word: str) -> float:
        """Normalized character-model probability of one word spelling."""
        self._ensure_spelling_cache_signature()
        cached = self._spelling_word_probability_cache.get(word)
        if cached is not None:
            return cached
        sequence = ([SPELL_BOW] * (self.max_order - 1)
                    + list(word.lower()) + [SPELL_EOW])
        log_probability = 0.0
        for index in range(self.max_order - 1, len(sequence)):
            context = tuple(sequence[max(0, index - self.max_order + 1):index])
            probability = self._spelling_char_probability(context, sequence[index])
            log_probability += math.log(max(probability, 1e-300))
        result = math.exp(max(log_probability, -745.0))
        self._spelling_word_probability_cache[word] = result
        return result

    def _spelling_oov_mass(self) -> float:
        """Global character spelling mass outside the current known vocabulary."""
        self._ensure_spelling_cache_signature()
        cached = self._spelling_oov_mass_cache.get(())
        if cached is None:
            known_mass = sum(
                self._spelling_word_probability(word)
                for word in self.vocab if word not in (BOUNDARY, UNK)
            )
            cached = max(1.0 - known_mass, 1e-300)
            self._spelling_oov_mass_cache[()] = cached
        return cached

    def _raw_oov_target_counts(self, context: tuple[str, ...]) -> tuple[Counter[str], int]:
        """Cached raw lexical targets collapsed into the current OOV class."""
        cached = self._raw_oov_counts_cache.get(context)
        if cached is None:
            counts = Counter({word: count for word, count in
                              self.raw_word_context_counts.get(context, {}).items()
                              if word not in (BOUNDARY, UNK) and self._map(word) == UNK})
            cached = counts, sum(counts.values())
            self._raw_oov_counts_cache[context] = cached
        return cached

    def _oov_spelling_prior_probability(
        self, raw_context: tuple[str, ...], surface_word: str,
    ) -> float:
        """Contextual raw rare-word interpolation over a global spelling prior."""
        context = tuple(word.lower() for word in raw_context)[-(self.max_order - 1):] \
            if self.max_order > 1 else ()
        prior = self._spelling_word_probability(surface_word) / self._spelling_oov_mass()
        weights = [self.observer.get_weight(self.word_level_nodes[order].id)
                   for order in range(self.max_order)] if self.observer else [0.5] * self.max_order
        for order in range(min(len(context), self.max_order - 1) + 1):
            suffix = context[-order:] if order else ()
            counts, total = self._raw_oov_target_counts(suffix)
            if total <= 0:
                continue
            evidence = total / (total + self.alpha)
            local_mass = weights[order] * evidence
            prior = (local_mass * counts.get(surface_word.lower(), 0) / total
                     + (1.0 - local_mass) * prior)
        return prior

    def surface_word_probability(self, context: Iterable[str], surface_word: str) -> float:
        """Open-vocabulary next-word probability, including exact OOV spellings.

        The Observer word distribution supplies probability mass for known
        words and the aggregate UNK class. The spelling LM distributes UNK
        mass over strings outside the current lexical vocabulary and can also
        contribute a calibrated character channel over all lexical strings.
        Sentence EOS retains its word-model probability.
        """
        raw_context = tuple(word.lower() for word in context)[-(self.max_order - 1):] \
            if self.max_order > 1 else ()
        mapped_context = tuple(self._map(word) for word in raw_context)
        target = self._map(surface_word.lower())
        base = self._weighted_probability(mapped_context, target,
                                          apply_character_mix=False)
        if surface_word == BOUNDARY:
            return base
        mix = self.character_word_mix
        if target != UNK:
            char_probability = self._spelling_word_probability(surface_word)
            eos_probability = self._weighted_probability(
                mapped_context, BOUNDARY, apply_character_mix=False,
            )
            return (1.0 - mix) * base + mix * (1.0 - eos_probability) * char_probability
        char_probability = self._spelling_word_probability(surface_word)
        eos_probability = self._weighted_probability(
            mapped_context, BOUNDARY, apply_character_mix=False,
        )
        contextual_spelling_prior = self._oov_spelling_prior_probability(raw_context, surface_word)
        return ((1.0 - mix) * base * contextual_spelling_prior
                + mix * (1.0 - eos_probability) * char_probability)

    def _decode_oov_spellings(
        self, k: int, *, min_word_length: int = 3, max_word_length: int = 24,
        max_expansions: int = 10_000,
    ) -> list[str]:
        """Best-first decode high-probability OOV strings from the spelling LM.

        Prefix probability is an upper bound on every completion below it, so
        once the best queued prefix cannot beat the kth completed spelling,
        the returned spellings are the top-k under the character model (unless
        the explicit expansion budget was reached first).
        """
        if (k < 1 or min_word_length < 1 or max_word_length < min_word_length
                or max_expansions < 1):
            raise ValueError("k, min_word_length, max_word_length and max_expansions must be positive and ordered")
        signature = self._spelling_signature()
        self._ensure_spelling_cache_signature()
        cache_key = (k, min_word_length, max_word_length, max_expansions)
        if cache_key in self._spelling_oov_candidate_cache:
            return list(self._spelling_oov_candidate_cache[cache_key])
        char_alphabet = [char for char in self.spelling_char_vocab
                         if char not in (SPELL_EOW, SPELL_UNK)]
        if not char_alphabet:
            return []
        known_words = set(self.vocab) - {BOUNDARY, UNK}
        initial_context = (SPELL_BOW,) * (self.max_order - 1)
        frontier = [(0.0, "", initial_context)]
        completed: dict[str, float] = {}
        expansions = 0
        while frontier and expansions < max_expansions:
            neg_log_probability, prefix, context = heapq.heappop(frontier)
            log_probability = -neg_log_probability
            if len(completed) >= k:
                kth_log_probability = sorted(completed.values(), reverse=True)[k - 1]
                if log_probability <= kth_log_probability:
                    break
            if len(prefix) >= max_word_length:
                continue
            expansions += 1
            for target in (*char_alphabet, SPELL_EOW):
                probability = self._spelling_char_probability(context, target)
                if probability <= 0.0:
                    continue
                child_log_probability = log_probability + math.log(probability)
                if target == SPELL_EOW:
                    if len(prefix) >= min_word_length and prefix not in known_words:
                        completed[prefix] = child_log_probability
                    continue
                next_context = ((*context, target)[-(self.max_order - 1):]
                                if self.max_order > 1 else ())
                heapq.heappush(frontier, (-child_log_probability, prefix + target, next_context))
        result = [word for word, _ in sorted(completed.items(),
                                             key=lambda item: (-item[1], item[0]))[:k]]
        self._spelling_oov_candidate_cache[cache_key] = tuple(result)
        return result

    def predict_surface_next_words(
        self, context: Iterable[str], k: int = 10, *, include_eos: bool = False,
        min_word_length: int = 3, max_word_length: int = 24,
        max_expansions: int = 10_000,
    ) -> list[dict[str, str | float | bool]]:
        """Rank known and character-decoded OOV next-word candidates.

        OOV candidates are decoded from the spelling LM, then scored by the
        normalized surface distribution used for held-out surface cross-entropy.
        This is candidate search, not a neural or encoder-based decoder.
        """
        if k < 1:
            raise ValueError("k must be positive")
        raw_context = [word.lower() for word in context]
        mapped_context = (tuple(self._map(word.lower()) for word in raw_context)
                          [-(self.max_order - 1):] if self.max_order > 1 else ())
        known_candidates = set(self._top_candidates(mapped_context, k)) - {BOUNDARY, UNK}
        if self.character_word_mix > 0.0:
            known_cache_key = (k, max_word_length)
            if known_cache_key not in self._spelling_known_candidate_cache:
                known_words = [word for word in self.vocab if word not in (BOUNDARY, UNK)]
                char_ranked = sorted(known_words,
                                     key=lambda word: (-self._spelling_word_probability(word), word))[:k]
                self._spelling_known_candidate_cache[known_cache_key] = tuple(char_ranked)
            known_candidates.update(self._spelling_known_candidate_cache[known_cache_key])
        for order in range(min(len(raw_context), self.max_order - 1) + 1):
            suffix = tuple(raw_context[-order:]) if order else ()
            raw_counts, _total = self._raw_oov_target_counts(suffix)
            raw_ranked = sorted(raw_counts, key=lambda word: (-raw_counts[word], word))[:k]
            known_candidates.update(raw_ranked)
        decoded_oov = self._decode_oov_spellings(
            k, min_word_length=min_word_length,
            max_word_length=max_word_length,
            max_expansions=max_expansions,
        )
        candidates = known_candidates | set(decoded_oov)
        if include_eos:
            candidates.add(BOUNDARY)
        scored = [{"word": word,
                   "probability": self.surface_word_probability(raw_context, word),
                   "is_oov": word not in self.word_to_index and word != BOUNDARY}
                  for word in candidates]
        return sorted(scored, key=lambda row: (-float(row["probability"]), str(row["word"])))[:k]

    def _weighted_probability(
        self, context: tuple[str, ...], target: str, *, character: bool = False,
        apply_character_mix: bool = True, apply_class_mix: bool = True,
    ) -> float:
        counts_table = self.char_counts if character else self.counts
        vocabulary = self.char_vocab if character else self.vocab
        alpha = self.char_alpha if character else self.alpha
        level_nodes = self.char_level_nodes if character else self.word_level_nodes
        flat_class_fallback = not character and self.class_nodes_enabled and self.class_lambda <= 0.0
        if flat_class_fallback:
            # The class-enabled learner explicitly compares against the flat
            # interpolated n-gram. If no class earns its place, use that expert
            # directly while retaining the trained Observer weights in state.
            probability = self._count_probability_for_context(context, target)
        elif not level_nodes or self.observer is None:
            if character:
                return self._char_distribution(context).get(target, 0.0)
            return self._distribution_for_context(context).get(target, 0.0)
        else:
            unigram = counts_table.get((), Counter())
            unigram_total = sum(unigram.values())
            probability = (unigram.get(target, 0) + alpha) / (unigram_total + alpha * len(vocabulary))
            unigram_weight = self.observer.get_weight(level_nodes[0].id)
            probability = unigram_weight * probability + (1.0 - unigram_weight) / len(vocabulary)
            for order in range(1, min(len(context), self.max_order - 1) + 1):
                counts = counts_table.get(context[-order:], Counter())
                total = sum(counts.values())
                if not total:
                    continue
                observer_weight = self.observer.get_weight(level_nodes[order].id)
                evidence_mass = total / (total + alpha)
                local_mass = observer_weight * evidence_mass
                probability = (local_mass * counts.get(target, 0) / total
                               + (1.0 - local_mass) * probability)
        if not character and apply_class_mix and self.class_lambda > 0:
            class_probability = self._class_mixed_probability(context, target, probability)
            if class_probability is not None:
                probability = class_probability
        if not character and apply_character_mix and self.character_word_mix > 0.0:
            if target == BOUNDARY:
                return probability
            eos_probability = self._weighted_probability(
                context, BOUNDARY, apply_character_mix=False,
            )
            if target == UNK:
                spelling_probability = self._spelling_oov_mass()
            else:
                spelling_probability = self._spelling_word_probability(target)
            probability = ((1.0 - self.character_word_mix) * probability
                           + self.character_word_mix * (1.0 - eos_probability) * spelling_probability)
        return probability

    def _character_word_prior(self) -> dict[str, float]:
        """Compose a normalized word-form prior from the character HFN model."""
        observer_weights = tuple(
            self.observer.get_weight(node.id) for _, node in sorted(self.char_level_nodes.items())
        ) if self.observer is not None else ()
        key = (self.char_alpha, observer_weights, self.max_order)
        if self._character_word_prior_cache is not None and self._character_word_prior_cache[0] == key:
            return self._character_word_prior_cache[1]

        counts_table, vocabulary = self.char_counts, self.char_vocab
        unigram = counts_table.get((), Counter())
        unigram_total = sum(unigram.values())
        denominator = unigram_total + self.char_alpha * len(vocabulary)

        def char_probability(context: tuple[str, ...], target: str) -> float:
            probability = (unigram.get(target, 0) + self.char_alpha) / denominator
            if observer_weights:
                weight = observer_weights[0]
                probability = weight * probability + (1.0 - weight) / len(vocabulary)
                for order in range(1, min(len(context), self.max_order - 1) + 1):
                    counts = counts_table.get(context[-order:], Counter())
                    total = sum(counts.values())
                    if not total:
                        continue
                    local_mass = observer_weights[order] * total / (total + self.char_alpha)
                    probability = (local_mass * counts.get(target, 0) / total
                                   + (1.0 - local_mass) * probability)
            return probability

        def spelling_score(word: str) -> float:
            sequence = [BOUNDARY] * (self.max_order - 1) + list(word.lower()) + [BOUNDARY]
            score = 0.0
            for i in range(self.max_order - 1, len(sequence)):
                target = sequence[i] if sequence[i] in self.char_vocab else UNK
                context = tuple(sequence[i - self.max_order + 1:i]) if self.max_order > 1 else ()
                context = tuple(char if char in self.char_vocab else UNK for char in context)
                score += math.log(max(char_probability(context, target), 1e-15))
            return score

        scores = {word: spelling_score(word) for word in self.vocab if word not in (BOUNDARY, UNK)}
        unknown_scores = [spelling_score(word) for word, count in self.word_counts.items()
                          if count < self.min_word_count]
        boundary_sequence = [BOUNDARY] * self.max_order
        context = tuple(boundary_sequence[:-1]) if self.max_order > 1 else ()
        boundary_score = math.log(max(
            char_probability(context, BOUNDARY), 1e-15
        ))
        scores[BOUNDARY] = boundary_score
        # UNK represents a class of unseen spellings. Use the geometric mean
        # score of training words collapsed into that class; with no collapsed
        # words, use the vocabulary-average spelling score.
        fallback = (sum(scores.values()) / len(scores)) if scores else boundary_score
        scores[UNK] = (sum(unknown_scores) / len(unknown_scores)) if unknown_scores else fallback
        peak = max(scores.values())
        masses = {word: math.exp(score - peak) for word, score in scores.items()}
        total = sum(masses.values()) or 1.0
        prior = {word: mass / total for word, mass in masses.items()}
        self._character_word_prior_cache = (key, prior)
        return prior

    def tune_character_word_mix(
        self, validation_documents: Iterable[str], *,
        max_events: int = 20_000,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict:
        """Select character-to-word composition strength on exact surface-form loss."""
        if max_events < 1:
            raise ValueError("max_events must be positive")
        documents = list(validation_documents)
        initial = self.character_word_mix
        before = self.score_surface_cross_entropy(documents)
        sample_before = self.score_surface_cross_entropy(documents, max_events=max_events)
        sweep = []
        candidates = sorted(set(CHARACTER_WORD_MIX_CANDIDATES) | {initial})
        for index, mix in enumerate(candidates, 1):
            self.character_word_mix = mix
            loss = self.score_surface_cross_entropy(documents, max_events=max_events)
            sweep.append({"mix": mix, "validation_surface_cross_entropy_nats": loss})
            if progress_callback:
                progress_callback({"completed_candidates": index,
                                   "total_candidates": len(candidates),
                                   "candidate_mix": mix,
                                   "validation_surface_cross_entropy_nats": loss})
        best = min(sweep, key=lambda row: (
            row["validation_surface_cross_entropy_nats"], abs(row["mix"] - initial), row["mix"]
        ))
        self.character_word_mix = best["mix"]
        after = self.score_surface_cross_entropy(documents)
        rolled_back = after > before + 1e-12
        if rolled_back:
            self.character_word_mix = initial
            after = before
        sample_after = min(row["validation_surface_cross_entropy_nats"] for row in sweep)
        selected = self.character_word_mix
        total_events = sum(
            len(tokenize(sentence)) + 1
            for text in documents for sentence in split_sentences(text)
        )
        return {"previous_mix": initial, "selected_mix": selected,
                "validation_surface_cross_entropy_before": before,
                "validation_surface_cross_entropy_after": after,
                "sample_validation_surface_cross_entropy_before": sample_before,
                "sample_validation_surface_cross_entropy_after": sample_after,
                "validation_events": total_events,
                "calibration_events": min(max_events, total_events),
                "rolled_back_on_full_validation": rolled_back,
                "selection_metric": "sampled_validation_exact_surface_word_cross_entropy_then_full_validation_check",
                "sweep": sweep}

    def _char_distribution(self, context: tuple[str, ...]) -> dict[str, float]:
        counts = self.char_counts.get(context, Counter())
        total = sum(counts.values())
        denom = total + self.char_alpha * len(self.char_vocab)
        return {char: (counts.get(char, 0) + self.char_alpha) / denom for char in self.char_vocab}

    def predict_proba(self, context: Iterable[str]) -> np.ndarray:
        mapped = tuple(self._map(w.lower()) for w in context)[-(self.max_order - 1):] if self.max_order > 1 else ()
        distribution = self._weighted_distribution(mapped)
        return np.asarray([distribution[word] for word in self.vocab])

    def _probability(self, context: tuple[str, ...], target: str) -> float:
        return self._weighted_probability(context, target)

    def _top_candidates(self, context: tuple[str, ...], k: int) -> frozenset[str]:
        cache_key = (context, k)
        if cache_key in self._candidate_cache:
            return self._candidate_cache[cache_key]
        candidates = set()
        while True:
            counts = self.counts.get(context, Counter())
            ranked = sorted(counts.items(), key=lambda item: (item[1], self.word_to_index[item[0]]), reverse=True)
            if ranked:
                cutoff = ranked[min(k, len(ranked)) - 1][1]
                candidates.update(word for word, value in ranked if value >= cutoff)
            if not context:
                break
            context = context[1:]
        result = frozenset(candidates)
        # UNK may have zero training counts at min_word_count=1, but remains a
        # legitimate smoothed vocabulary item and must participate in ranking.
        result = frozenset(set(result) | {UNK})
        self._candidate_cache[cache_key] = result
        return result
