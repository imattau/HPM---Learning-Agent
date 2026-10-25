"""Encoder-free character/word/phrase next-unit learning pilot.

The model stores empirical conditional distributions in HFN nodes.  It is a
small, inspectable count learner (not a neural language model); the experiment
exists to test whether HPM's hierarchy adds predictive value over n-grams.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hpm_ai_v2.training.l5_language_meta import PrequentialL5Mixer

SPELL_BOW = "<bow>"
SPELL_EOW = "</w>"
SPELL_UNK = "<unk-char>"
MAX_FIT_WORKERS = 4
CLASS_MAX_SEEN_REGRESSION_NATS = 0.05
CLASS_RESERVED_UNSEEN_MASS = 0.04
CLASS_MAX_MEMBERS = 64
CLASS_MIN_COHESION = 0.10
CLASS_MIN_VALIDATION_PAIRS = 1
CLASS_LAMBDA_GRID = (0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0)
CHARACTER_WORD_MIX_CANDIDATES = (0.0, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5)
CLASS_MIX_CACHE_MAX_ENTRIES = 32
CLASS_MIX_MASS_CACHE_MAX_ENTRIES = 4096

from hpm_ai_v2.training.language_constants import BOUNDARY, TOKEN_RE, UNK
from hpm_ai_v2.training.language_corpus import (
    _batches, _count_raw_text_batch, read_jsonl, split_documents, split_sentences, tokenize,
)
from hpm_ai_v2.training.language_l5_mixin import LanguageL5Mixin as _LanguageL5Mixin
from hpm_ai_v2.training.language_validation_workers import (
    _run_metric_jobs, _run_validation_sweeps,
)

class HierarchicalNextUnitModel(_LanguageL5Mixin):
    """Count learner whose predictive contexts and lexical forms are HFN nodes."""

    def __init__(self, max_order: int = 3, alpha: float = .1, min_word_count: int = 1,
                 *, class_nodes: bool = False, class_tau: float | None = 0.55,
                 class_lambda: float = 0.5, class_min_observations: int = 3,
                 class_profile_limit: int = 1500, class_follower_limit: int = 2048):
        if max_order < 1 or alpha <= 0 or min_word_count < 1:
            raise ValueError("max_order and min_word_count must be positive and alpha must be > 0")
        self.max_order, self.alpha, self.min_word_count = max_order, alpha, min_word_count
        if class_tau is not None and (not math.isfinite(class_tau) or class_tau <= 0):
            raise ValueError("class_tau must be positive or None")
        if not math.isfinite(class_lambda) or not 0.0 <= class_lambda <= 1.0:
            raise ValueError("class_lambda must be in [0, 1]")
        if class_min_observations < 2:
            raise ValueError("class_min_observations must be at least 2")
        if class_profile_limit < 2 or class_follower_limit < 2:
            raise ValueError("class profile and follower limits must be at least 2")
        self.class_nodes_enabled = bool(class_nodes)
        self.class_tau = class_tau
        self.class_lambda = float(class_lambda)
        self.class_min_observations = int(class_min_observations)
        self.class_profile_limit = int(class_profile_limit)
        self.class_follower_limit = int(class_follower_limit)
        self.class_labels: dict[str, str] = {}
        self.class_lambdas: dict[str, float] = {}
        self.class_profiles: dict[str, np.ndarray] = {}
        self.class_profile_variances: dict[str, np.ndarray] = {}
        self.class_profile_features: list[str] = []
        self._current_profile_features: list[str] = []
        self.class_context_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.class_node_registry: dict[str, HFN] = {}
        self._class_membership_frozen = False
        self.class_selection: dict | None = None
        self.char_alpha = alpha
        self.character_word_mix = 0.0
        self._character_word_prior_cache: tuple[tuple, dict[str, float]] | None = None
        self.word_counts: Counter[str] = Counter()
        self.counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.raw_word_context_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self._raw_oov_counts_cache: dict[tuple[str, ...], tuple[Counter[str], int]] = {}
        self._raw_counts_complete = True
        self.vocab: list[str] = []
        self.nodes: dict[str, HFN] = {}
        self.char_nodes: dict[str, HFN] = {}
        self.word_nodes: dict[str, HFN] = {}
        self.context_nodes: dict[tuple[str, ...], HFN] = {}
        self.class_context_nodes: dict[tuple[str, ...], HFN] = {}
        self._class_mix_mass_cache: OrderedDict[tuple, tuple[float, float, float]] = OrderedDict()
        self.phrase_nodes: dict[tuple[str, ...], HFN] = {}
        self.sentence_nodes: dict[str, HFN] = {}
        self.sentence_counts: Counter[str] = Counter()
        self.pattern_forest: Forest | None = None
        self.observer: Observer | None = None
        self.word_level_nodes: dict[int, HFN] = {}
        self.char_level_nodes: dict[int, HFN] = {}
        self.char_context_nodes: dict[tuple[str, ...], HFN] = {}
        self.char_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.char_vocab: list[str] = []
        self.spelling_char_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.spelling_char_vocab: list[str] = []
        self._spelling_cache_signature: tuple | None = None
        self._spelling_word_probability_cache: dict[str, float] = {}
        self._spelling_oov_mass_cache: dict[tuple[str, ...], float] = {}
        self._spelling_oov_candidate_cache: dict[tuple, tuple[str, ...]] = {}
        self._spelling_known_candidate_cache: dict[tuple, tuple[str, ...]] = {}
        self.training_manifest: dict | None = None
        self.internal_observer_calibration: dict | None = None
        self._class_validation_documents: list[str] | None = None
        # Exact-text identity lets checkpoint continuation ignore epoch replay
        # instead of treating already-seen examples as independent evidence.
        self._training_document_hashes: set[str] = set()
        self._candidate_cache: dict[tuple[tuple[str, ...], int], frozenset[str]] = {}
        self.l5_language_state: dict | None = None
        self.l5_calibration_document_hashes: set[str] = set()

    def fit(
        self, documents: Iterable[str], workers: int = 1, *,
        progress_callback: Callable[[dict], None] | None = None,
        _skip_observer_validation_calibration: bool = False,
    ) -> "HierarchicalNextUnitModel":
        if workers < 1:
            raise ValueError("workers must be at least 1")
        def report(stage: str, **details) -> None:
            if progress_callback:
                progress_callback({"fit_stage": stage, **details})

        self._candidate_cache.clear()
        self._character_word_prior_cache = None
        self._class_mix_mass_cache.clear()
        if hasattr(self, "_class_mix_distribution_cache"):
            self._class_mix_distribution_cache.clear()
        texts = list(documents)
        if self.class_nodes_enabled and not _skip_observer_validation_calibration:
            # Keep a document-grouped calibration slice out of the initial
            # counts, tune Observer reliability on genuinely unseen documents,
            # then add those documents through the stable-membership
            # continuation path and restore the calibrated weights.
            text_groups: dict[str, list[str]] = defaultdict(list)
            for text in texts:
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                text_groups[digest].append(text)
            ordered_groups = sorted(
                text_groups.items(),
                key=lambda item: hashlib.sha256(
                    f"observer-calibration-v1:{item[0]}".encode("utf-8")
                ).digest(),
            )
            if len(ordered_groups) >= 10:
                calibration_group_count = max(1, round(len(ordered_groups) * 0.1))
                # Build a document-heldout class-validation set around whole
                # adjacent-pair occurrences. A random document split can
                # contain zero unseen pairs on restricted grammars (as in the
                # composition testbed), while hashing edges out of count
                # tables leaks the surrounding document. Instead, select
                # pairs first, then keep every document containing a selected
                # pair out of the fit partition. Profiles only see fit docs.
                pair_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
                pair_followers: dict[str, set[str]] = defaultdict(set)
                for group_id, group_texts in text_groups.items():
                    for text in group_texts:
                        for sentence in split_sentences(text):
                            sequence = tokenize(sentence) + [BOUNDARY]
                            for left, right in zip(sequence, sequence[1:]):
                                pair_groups[(left, right)].add(group_id)
                                pair_followers[left].add(right)
                # Prioritize candidate edges whose left word has a genuine
                # follower profile. Holding out a one-outcome context cannot
                # validate a learned distributional class and wastes scarce
                # document holdout capacity on structural boilerplate.
                candidate_pairs = [
                    pair for pair in pair_groups
                    if len(pair_followers[pair[0]]) >= 2
                ]
                ranked_pairs = sorted(
                    candidate_pairs or pair_groups,
                    key=lambda pair: (
                        len(pair_groups[pair]),
                        hashlib.sha256(
                            f"class-pair-validation-v1:{pair[0]}\0{pair[1]}".encode("utf-8")
                        ).digest(),
                    ),
                )
                target_pair_count = max(1, round(len(ranked_pairs) * 0.1))
                max_validation_groups = max(1, int(len(ordered_groups) * 0.2))
                selected_pairs: set[tuple[str, str]] = set()
                selected_group_ids: set[str] = set()
                selected_left_tokens: set[str] = set()
                for pair in ranked_pairs:
                    # Spread held-out edges across different context words
                    # before taking a second edge from the same context. This
                    # gives each candidate class member a chance to provide
                    # independent validation evidence.
                    if pair[0] in selected_left_tokens:
                        continue
                    proposed_groups = selected_group_ids | pair_groups[pair]
                    if len(proposed_groups) > max_validation_groups:
                        continue
                    selected_pairs.add(pair)
                    selected_left_tokens.add(pair[0])
                    selected_group_ids = proposed_groups
                    if len(selected_pairs) >= target_pair_count:
                        break
                if len(selected_pairs) < target_pair_count:
                    for pair in ranked_pairs:
                        # Keep the holdout spread across contexts even when
                        # the document cap prevents the first pass from
                        # reaching its target. Holding every follower of one
                        # word can erase the very profile needed to evaluate
                        # that word's class abstraction.
                        if pair in selected_pairs or pair[0] in selected_left_tokens:
                            continue
                        proposed_groups = selected_group_ids | pair_groups[pair]
                        if len(proposed_groups) > max_validation_groups:
                            continue
                        selected_pairs.add(pair)
                        selected_left_tokens.add(pair[0])
                        selected_group_ids = proposed_groups
                        if len(selected_pairs) >= target_pair_count:
                            break
                # Include a little ordinary document-heldout data when the
                # pair-selected groups alone are smaller than a 10% slice.
                ordinary_target = max(1, round(len(ordered_groups) * 0.1))
                for group_id, _group in ordered_groups:
                    if len(selected_group_ids) >= ordinary_target:
                        break
                    selected_group_ids.add(group_id)
                class_validation_groups = [
                    item for item in ordered_groups if item[0] in selected_group_ids
                ]
                remaining_groups = [
                    item for item in ordered_groups if item[0] not in selected_group_ids
                ]
                # Reserve independent document groups for order-level
                # calibration and class reinforcement. Reusing the
                # pair-holdout documents for reinforcement would make the
                # class's final weight adapt to the same outcomes that chose
                # its tau/lambda in _validation_class_choice.
                max_calibration_groups = max(0, len(remaining_groups) - 2)
                if max_calibration_groups >= 2:
                    calibration_group_count = min(
                        max(2, calibration_group_count), max_calibration_groups
                    )
                else:
                    calibration_group_count = 0
                calibration_groups = (
                    remaining_groups[-calibration_group_count:]
                    if calibration_group_count else []
                )
                fit_groups = (
                    remaining_groups[:-calibration_group_count]
                    if calibration_group_count else remaining_groups
                )
                if len(fit_groups) < 2:
                    fit_groups, class_validation_groups, calibration_groups = [], [], []
                fit_documents = [text for _, group in fit_groups for text in group]
                class_validation_documents = [
                    text for _, group in class_validation_groups for text in group
                ]
                order_calibration_groups = calibration_groups[::2]
                class_reinforcement_groups = calibration_groups[1::2]
                calibration_documents = [
                    text for _, group in order_calibration_groups for text in group
                ]
                class_reinforcement_documents = [
                    text for _, group in class_reinforcement_groups for text in group
                ]
                if (fit_documents and calibration_documents
                        and class_reinforcement_documents and class_validation_documents):
                    self._class_validation_documents = class_validation_documents
                    report("internal_observer_calibration", state="fit_partition",
                           fit_documents=len(fit_documents),
                           calibration_documents=len(calibration_documents),
                           calibration_groups=len(order_calibration_groups),
                           class_reinforcement_documents=len(class_reinforcement_documents),
                           class_reinforcement_groups=len(class_reinforcement_groups),
                           class_validation_documents=len(class_validation_documents),
                           class_validation_groups=len(class_validation_groups),
                           selected_validation_pairs=len(selected_pairs))
                    # Establish the word-level experts before selecting class
                    # weights. Class selection must score the same Observer-
                    # weighted predictor that will be used after fitting.
                    self.class_nodes_enabled = False
                    self.fit(fit_documents, workers=workers,
                             progress_callback=progress_callback,
                             _skip_observer_validation_calibration=True)
                    self.class_nodes_enabled = True
                    calibration = self.tune_observer_weights(calibration_documents)
                    calibrated_weights = {
                        order: self.observer.get_weight(node.id)
                        for order, node in self.word_level_nodes.items()
                    }
                    self._fit_class_nodes_from_counts(refit=True)
                    if any(self.class_lambdas.values()):
                        self._build_hfn_nodes(progress_callback=progress_callback)
                        self._fit_pattern_observer(progress_callback=progress_callback)
                        for order, node in self.word_level_nodes.items():
                            if order in calibrated_weights:
                                self.observer.set_weight(node.id, calibrated_weights[order])
                    # Reinforce class nodes on a separate document group from
                    # both class selection and word-order calibration. Score
                    # these unseen class events before returning them to the
                    # training counts below.
                    class_calibration = self._reinforce_class_observer_weights(
                        class_reinforcement_documents
                    )
                    if progress_callback:
                        progress_callback({
                            "fit_stage": "internal_observer_calibration",
                            "state": "complete",
                            "fit_documents": len(fit_documents),
                            "calibration_documents": len(calibration_documents),
                            "class_reinforcement_documents": len(class_reinforcement_documents),
                            "validation_cross_entropy_before": calibration[
                                "validation_cross_entropy_before"
                            ],
                            "validation_cross_entropy_after": calibration[
                                "validation_cross_entropy_after"
                            ],
                            "selected_weights": {
                                str(order): weight
                                for order, weight in calibrated_weights.items()
                            },
                            "class_observer_training": class_calibration,
                        })
                    self.partial_fit(
                        [*class_validation_documents, *calibration_documents,
                         *class_reinforcement_documents], workers=workers,
                        _preserve_batch_frequency=True,
                    )
                    class_refit = None
                    if self.class_tau is not None and any(self.class_lambdas.values()):
                        class_refit = self._refit_selected_classes_on_full_training_counts()
                        self._build_hfn_nodes(progress_callback=progress_callback)
                        self._fit_pattern_observer(progress_callback=progress_callback)
                        for order, node in self.word_level_nodes.items():
                            if order in calibrated_weights:
                                self.observer.set_weight(node.id, calibrated_weights[order])
                        for class_id, weight in self.class_lambdas.items():
                            if class_id in self.class_node_registry:
                                self.observer.set_weight(class_id, weight)
                    for order, node in self.word_level_nodes.items():
                        if order in calibrated_weights:
                            self.observer.set_weight(node.id, calibrated_weights[order])
                    self.internal_observer_calibration = {
                        "fit_documents": len(fit_documents),
                        "calibration_documents": len(calibration_documents),
                        "calibration_groups": len(order_calibration_groups),
                        "class_reinforcement_documents": len(class_reinforcement_documents),
                        "class_reinforcement_groups": len(class_reinforcement_groups),
                        "class_validation_documents": len(class_validation_documents),
                        "class_validation_groups": len(class_validation_groups),
                        "selected_validation_pairs": len(selected_pairs),
                        "validation_cross_entropy_before": calibration[
                            "validation_cross_entropy_before"
                        ],
                        "validation_cross_entropy_after": calibration[
                            "validation_cross_entropy_after"
                        ],
                        "selected_weights": {
                            str(order): weight
                            for order, weight in calibrated_weights.items()
                        },
                        "class_observer_training": class_calibration,
                        "class_final_refit": class_refit,
                    }
                    self._class_validation_documents = None
                    return self
        self._training_document_hashes = {
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
        }
        self.counts = defaultdict(Counter)
        self.raw_word_context_counts = defaultdict(Counter)
        self._raw_counts_complete = True
        self.char_counts = defaultdict(Counter)
        self.sentence_counts = Counter()
        # Parsing and character n-gram counting are document-local. Run those
        # in bounded processes for larger corpora, then merge deterministically
        # in this process before mutating the shared HFN graph.
        batches = list(_batches(texts))
        report("counting_documents", state="running", documents=len(texts),
               batches=len(batches), workers=workers)
        use_processes = (workers > 1 and len(texts) >= 32
                         and sum(map(len, texts)) >= 20_000
                         and "fork" in multiprocessing.get_all_start_methods())
        if use_processes:
            with ProcessPoolExecutor(max_workers=min(workers, len(batches), MAX_FIT_WORKERS),
                                     mp_context=multiprocessing.get_context("fork")) as pool:
                raw_results = []
                for index, result in enumerate(pool.map(
                        _count_raw_text_batch, batches, [self.max_order] * len(batches)), 1):
                    raw_results.append(result)
                    report("counting_documents", state="running", completed_batches=index,
                           total_batches=len(batches), workers=workers)
        else:
            raw_results = []
            for index, batch in enumerate(batches, 1):
                raw_results.append(_count_raw_text_batch(batch, self.max_order))
                report("counting_documents", state="running", completed_batches=index,
                       total_batches=len(batches), workers=workers)
        sentence_docs = [sentences for tokenized, _, _ in raw_results for sentences in tokenized]
        self.word_counts = Counter()
        self.char_counts = defaultdict(Counter)
        for _, batch_words, batch_chars in raw_results:
            self.word_counts.update(batch_words)
            for context, targets in batch_chars.items():
                self.char_counts[context].update(targets)
        self.vocab = sorted({w if c >= self.min_word_count else UNK for w, c in self.word_counts.items()} | {BOUNDARY, UNK})
        self.word_to_index = {w: i for i, w in enumerate(self.vocab)}
        self.char_vocab = sorted({ch for text in texts for ch in text.lower()} | {BOUNDARY, UNK})
        for sentences in sentence_docs:
            for doc in sentences:
                sequence = list(doc) + [BOUNDARY]
                padded = [BOUNDARY] * (self.max_order - 1) + sequence
                for i in range(self.max_order - 1, len(padded)):
                    target = padded[i]
                    for order in range(self.max_order):
                        context = tuple(padded[i-order:i]) if order else ()
                        self.raw_word_context_counts[context][target] += 1
                sentence_text = " ".join(doc)
                if sentence_text:
                    self.sentence_counts[sentence_text] += 1
        self._rebuild_mapped_word_counts()
        report("count_tables", state="complete", vocabulary=len(self.vocab),
               word_contexts=len(self.raw_word_context_counts),
               character_contexts=len(self.char_counts))
        if self.class_nodes_enabled:
            report("class_validation", state="running", words=len(self.word_counts))
            self._fit_class_nodes_from_counts(refit=True)
            report("class_validation", state="complete", selected_tau=self.class_tau,
                   selected_lambda=self.class_lambda,
                   classes=len(set(self.class_labels.values())))
        self._rebuild_spelling_char_model()
        report("hfn_node_build", state="running")
        self._build_hfn_nodes(progress_callback=progress_callback)
        report("observer_initialization", state="running", hfn_nodes=len(self.nodes))
        self._fit_pattern_observer(progress_callback=progress_callback)
        report("observer_initialization", state="complete", hfn_nodes=len(self.nodes),
               forest_nodes=len(self.pattern_forest) if self.pattern_forest else 0)
        return self

    def _rebuild_mapped_word_counts(self) -> None:
        """Map raw lexical n-gram sufficient statistics through the current vocab."""
        self._raw_oov_counts_cache.clear()
        self.vocab = sorted({word if count >= self.min_word_count else UNK
                             for word, count in self.word_counts.items()} | {BOUNDARY, UNK})
        self.word_to_index = {word: index for index, word in enumerate(self.vocab)}
        self.counts = defaultdict(Counter)
        for raw_context, targets in self.raw_word_context_counts.items():
            context = tuple(self._map(word) for word in raw_context)
            for target, count in targets.items():
                self.counts[context][self._map(target)] += count

    def _rebuild_spelling_char_model(self) -> None:
        """Build a normalized character spelling LM from lexical training counts.

        Unlike the sentence character model, this model predicts a word's
        spelling from a word-start marker through an explicit end marker. Its
        distribution covers words outside the current finite vocabulary too.
        """
        self.spelling_char_counts = defaultdict(Counter)
        characters = set()
        for word, frequency in self.word_counts.items():
            if word in (BOUNDARY, UNK) or frequency <= 0:
                continue
            characters.update(word.lower())
            sequence = ([SPELL_BOW] * (self.max_order - 1)
                        + list(word.lower()) + [SPELL_EOW])
            for index in range(self.max_order - 1, len(sequence)):
                target = sequence[index]
                for order in range(self.max_order):
                    context = tuple(sequence[index - order:index]) if order else ()
                    self.spelling_char_counts[context][target] += frequency
        self.spelling_char_vocab = sorted(characters | {SPELL_EOW, SPELL_UNK})
        self._spelling_cache_signature = None
        self._spelling_word_probability_cache.clear()
        self._spelling_oov_mass_cache.clear()
        self._spelling_oov_candidate_cache.clear()
        self._spelling_known_candidate_cache.clear()

    def partial_fit(
        self, documents: Iterable[str], workers: int = 1, *,
        observer_replay_documents: Iterable[str] = (),
        class_validation_documents: Iterable[str] = (),
        min_word_count: int | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        _preserve_batch_frequency: bool = False,
    ) -> "HierarchicalNextUnitModel":
        """Continue the count learner with new training documents.

        New documents add raw lexical and character sufficient statistics to
        the existing checkpoint. The HFN graph is rebuilt from accumulated
        counts and Observer level reliability is recalculated; validation must
        then retune weights and the character-to-word mix before test scoring.
        Pass training documents only. Parsing stays serial after a graph exists
        so worker forks do not inherit a populated Forest/Observer state.
        """
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if min_word_count is not None and min_word_count < 1:
            raise ValueError("min_word_count must be at least 1")
        if not self._raw_counts_complete:
            raise ValueError(
                "checkpoint lacks raw lexical context counts; refit from source training data before continuing"
            )
        supplied_texts = list(documents)
        class_validation_texts = [text for text in class_validation_documents if text.strip()]
        replay_texts = [text for text in observer_replay_documents if text.strip()]
        if not supplied_texts:
            return self
        supplied_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest()
                           for text in supplied_texts]
        previous_hashes = set(self._training_document_hashes)
        validation_hashes = {hashlib.sha256(text.encode("utf-8")).hexdigest()
                             for text in class_validation_texts}
        if validation_hashes & (set(supplied_hashes) | previous_hashes):
            raise ValueError("class validation documents must be held out from all HFN training data")
        seen_hashes = set(previous_hashes)
        texts = []
        new_hashes = []
        for text, digest in zip(supplied_texts, supplied_hashes):
            if (digest not in seen_hashes
                    or (_preserve_batch_frequency and digest not in previous_hashes)):
                texts.append(text)
                new_hashes.append(digest)
                seen_hashes.add(digest)
        if not texts:
            if self.training_manifest is not None:
                replay_batches = self.training_manifest.setdefault("replay_training_batches", [])
                replay_batches.append({"documents": len(supplied_texts), "new_documents": 0,
                                       "unique_text_sha256": sorted(set(supplied_hashes))})
            if progress_callback:
                progress_callback({"stage": "replay_skipped", "documents": len(supplied_texts),
                                   "new_documents": 0, "hfn_nodes": len(self.nodes)})
            return self
        if hasattr(self, "_class_mix_distribution_cache"):
            self._class_mix_distribution_cache.clear()
        self._class_mix_mass_cache.clear()
        previous_class_weights = {
            class_id: self.observer.get_weight(class_id)
            for class_id in self.class_node_registry if self.observer is not None
        }
        def progress(stage: str, **details) -> None:
            if progress_callback:
                progress_callback({"stage": stage, "documents": len(texts), **details})

        batches = list(_batches(texts))
        use_processes = (self.pattern_forest is None and workers > 1 and len(texts) >= 32
                         and sum(map(len, texts)) >= 20_000
                         and "fork" in multiprocessing.get_all_start_methods())
        progress("count_new_document_statistics", workers=(min(workers, len(batches), MAX_FIT_WORKERS)
                                                            if use_processes else 1))
        if use_processes:
            with ProcessPoolExecutor(max_workers=min(workers, len(batches), MAX_FIT_WORKERS),
                                     mp_context=multiprocessing.get_context("fork")) as pool:
                raw_results = list(pool.map(_count_raw_text_batch, batches,
                                            [self.max_order] * len(batches)))
        else:
            raw_results = [_count_raw_text_batch(batch, self.max_order) for batch in batches]
        progress("merge_raw_statistics", batches=len(raw_results))
        sentence_docs = [sentences for tokenized, _, _ in raw_results for sentences in tokenized]
        for _, batch_words, batch_chars in raw_results:
            self.word_counts.update(batch_words)
            for context, targets in batch_chars.items():
                self.char_counts[context].update(targets)
        if min_word_count is not None:
            self.min_word_count = min_word_count
        self._training_document_hashes.update(new_hashes)
        self.char_vocab = sorted(set(self.char_vocab) | {char for text in texts for char in text.lower()} | {BOUNDARY, UNK})
        for sentences in sentence_docs:
            for sentence in sentences:
                sequence = list(sentence) + [BOUNDARY]
                padded = [BOUNDARY] * (self.max_order - 1) + sequence
                for index in range(self.max_order - 1, len(padded)):
                    target = padded[index]
                    for order in range(self.max_order):
                        context = tuple(padded[index-order:index]) if order else ()
                        self.raw_word_context_counts[context][target] += 1
                sentence_text = " ".join(sentence)
                if sentence_text:
                    self.sentence_counts[sentence_text] += 1
        self._rebuild_mapped_word_counts()
        if self.class_nodes_enabled:
            self._fit_class_nodes_from_counts(refit=not self._class_membership_frozen)
        self._rebuild_spelling_char_model()
        self._candidate_cache.clear()
        self._character_word_prior_cache = None
        self._class_mix_mass_cache.clear()
        progress("rebuild_hfn_graph", raw_contexts=len(self.raw_word_context_counts),
                 mapped_contexts=len(self.counts))
        self._build_hfn_nodes()
        progress("recompute_observer_reliability", hfn_nodes=len(self.nodes))
        self._fit_pattern_observer()
        for class_id, weight in previous_class_weights.items():
            if class_id in self.class_node_registry:
                self.observer.set_weight(class_id, weight)
        progress("train_observer_weights", objective="training_leave_one_event_out_next_unit_cross_entropy")
        word_training = self.train_observer_weights(
            [*texts, *replay_texts], character=False,
            progress_callback=lambda details: progress("train_observer_weights", **details),
        )
        character_training = self.train_observer_weights(
            [*texts, *replay_texts], character=True,
            progress_callback=lambda details: progress("train_observer_weights", **details),
        )
        class_validation_training = (
            self._reinforce_class_observer_weights(class_validation_texts)
            if class_validation_texts else {"events": 0, "updated_classes": 0, "weights": {}}
        )
        if class_validation_texts:
            for class_id, weight in class_validation_training["weights"].items():
                self.class_lambdas[class_id] = 0.0 if weight < 0.005 else weight
            self.class_lambda = max(self.class_lambdas.values(), default=0.0)
        if self.training_manifest is not None:
            batches_manifest = self.training_manifest.setdefault("incremental_training_batches", [])
            batches_manifest.append({
                "documents": len(texts),
                "observer_replay_documents": len(replay_texts),
                "min_word_count": self.min_word_count,
                "characters": sum(map(len, texts)),
                "text_sha256": hashlib.sha256("\n".join(sorted(texts)).encode()).hexdigest(),
                "observer_training": {"word": word_training, "character": character_training},
                "class_observer_validation": class_validation_training,
            })
            if len(texts) != len(supplied_texts):
                replay_batches = self.training_manifest.setdefault("replay_training_batches", [])
                replay_batches.append({"documents": len(supplied_texts), "new_documents": len(texts),
                                       "unique_text_sha256": sorted(set(supplied_hashes))})
        progress("complete", hfn_nodes=len(self.nodes), observer_levels=2 * self.max_order,
                 observer_training={"word": word_training, "character": character_training})
        return self

    def recover_raw_training_contexts(
        self, documents: Iterable[str], workers: int = 1,
    ) -> dict[str, int]:
        """Reconstruct continuation statistics for a legacy checkpoint exactly.

        The supplied texts must be the complete original training partition.
        Word counts, character contexts, sentence counts, and mapped contexts
        are checked against the checkpoint before continuation is enabled.
        """
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if self._raw_counts_complete:
            return {"raw_contexts": len(self.raw_word_context_counts), "recovered": 0}
        texts = list(documents)
        batches = list(_batches(texts))
        use_processes = (workers > 1 and len(texts) >= 32
                         and sum(map(len, texts)) >= 20_000
                         and "fork" in multiprocessing.get_all_start_methods())
        if use_processes:
            with ProcessPoolExecutor(max_workers=min(workers, len(batches), MAX_FIT_WORKERS),
                                     mp_context=multiprocessing.get_context("fork")) as pool:
                results = list(pool.map(_count_raw_text_batch, batches,
                                        [self.max_order] * len(batches)))
        else:
            results = [_count_raw_text_batch(batch, self.max_order) for batch in batches]
        word_counts: Counter[str] = Counter()
        char_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        sentences = [rows for tokenized, _, _ in results for rows in tokenized]
        for _, batch_words, batch_chars in results:
            word_counts.update(batch_words)
            for context, targets in batch_chars.items():
                char_counts[context].update(targets)
        sentence_counts = Counter(" ".join(row) for rows in sentences for row in rows if row)
        if word_counts != self.word_counts or char_counts != self.char_counts or sentence_counts != self.sentence_counts:
            raise ValueError("provided texts do not exactly reproduce checkpoint training statistics")

        raw_contexts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        for rows in sentences:
            for row in rows:
                sequence = list(row) + [BOUNDARY]
                padded = [BOUNDARY] * (self.max_order - 1) + sequence
                for index in range(self.max_order - 1, len(padded)):
                    target = padded[index]
                    for order in range(self.max_order):
                        context = tuple(padded[index-order:index]) if order else ()
                        raw_contexts[context][target] += 1
        rebuilt: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        for raw_context, targets in raw_contexts.items():
            mapped_context = tuple(self._map(word) for word in raw_context)
            for target, count in targets.items():
                rebuilt[mapped_context][self._map(target)] += count
        if rebuilt != self.counts:
            raise ValueError("provided texts do not reproduce checkpoint mapped contexts")
        self.raw_word_context_counts = raw_contexts
        self._raw_counts_complete = True
        self._training_document_hashes = {
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
        }
        if self.training_manifest is not None:
            self.training_manifest["raw_training_contexts_recovered"] = True
        return {"raw_contexts": len(raw_contexts), "recovered": 1,
                "documents": len(texts), "characters": sum(map(len, texts))}

    def _map(self, token: str) -> str:
        if token in (BOUNDARY, UNK):
            return token
        return token if self.word_counts[token] >= self.min_word_count else UNK

    @staticmethod
    def _safe_id(prefix: str, text: str) -> str:
        return prefix + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _project_follower_profile(profile: np.ndarray, features: list[str]) -> np.ndarray:
        """Deterministically embed follower profiles into the Forest's D=4 space.

        A signed feature hash makes projection stable for a token across training
        rounds and independent of the current vocabulary's ordering.
        """
        projected = np.zeros(4, dtype=float)
        for feature, value in zip(features, profile):
            if value == 0:
                continue
            digest = hashlib.sha256(f"hpm-language-profile-projection-v1:{feature}".encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % len(projected)
            sign = 1.0 if digest[4] & 1 else -1.0
            projected[bucket] += sign * float(value)
        return projected

    def _profiles_from_counts(self, raw_counts: dict) -> dict[str, np.ndarray]:
        """IDF-weighted follower profiles conditioned on the preceding role cue.

        When the forest has at least a two-token context, the profile feature
        is ``(word_before_context_word, next_word)``. This prevents a word
        used in distinct positions (for example a subject noun before a verb
        and an object noun before sentence end) from having its role signals
        collapsed into one unconditional follower vector.
        """
        followers: dict[str, Counter] = defaultdict(Counter)
        for context, targets in raw_counts.items():
            if len(context) == 2 and context[-1] not in (BOUNDARY, UNK):
                role_cue = context[-2]
                for target, count in targets.items():
                    followers[context[-1]][f"{role_cue}\0{target}"] += count
        # Preserve order-2 models' original behavior, where there is no
        # preceding role cue available to condition the follower profile.
        if not followers:
            for context, targets in raw_counts.items():
                if len(context) == 1 and context[0] not in (BOUNDARY, UNK):
                    followers[context[0]].update(targets)
        eligible = {word: counts for word, counts in followers.items()
                    if sum(counts.values()) >= self.class_min_observations}
        if len(eligible) > self.class_profile_limit:
            eligible = dict(sorted(eligible.items(), key=lambda item: (-sum(item[1].values()), item[0]))[
                            :self.class_profile_limit])
        if not eligible:
            self._current_profile_features = []
            return {}
        follower_totals = Counter()
        for counts in eligible.values():
            follower_totals.update(counts)
        target_vocab = [word for word, _ in sorted(follower_totals.items(),
                                                    key=lambda item: (-item[1], item[0]))[
                        :self.class_follower_limit]]
        self._current_profile_features = list(target_vocab)
        target_index = {word: index for index, word in enumerate(target_vocab)}
        global_counts = Counter()
        for counts in eligible.values():
            global_counts.update({word: count for word, count in counts.items()
                                  if word in target_index})
        global_total = sum(global_counts.values())
        global_prob = np.asarray([(global_counts.get(word, 0) + 1e-9) /
                                  (global_total + 1e-9 * len(target_vocab))
                                  for word in target_vocab])
        profiles = {}
        for word, counts in eligible.items():
            retained = {target: count for target, count in counts.items() if target in target_index}
            total = sum(retained.values())
            if not total:
                continue
            observed = np.zeros(len(target_vocab), dtype=float)
            for target, count in retained.items():
                observed[target_index[target]] = count / total
            pmi = np.maximum(0.0, np.log((observed + 1e-12) / global_prob))
            profiles[word] = pmi
        df = np.sum(np.asarray([profile > 0 for profile in profiles.values()]), axis=0)
        idf = np.log((len(profiles) + 1.0) / (df + 1.0))
        weighted = {word: profile * idf for word, profile in profiles.items()}
        # Single-follower words mostly encode sentence templates (for example
        # every synthetic verb followed by "a"), not a useful lexical class.
        return {word: profile for word, profile in weighted.items()
                if np.count_nonzero(profile > 1e-12) >= 2}

    @staticmethod
    def _average_link_clusters(profiles: dict[str, np.ndarray], tau: float | None) -> list[list[str]]:
        words = sorted(profiles)
        if tau is None or len(words) < 2:
            return [[word] for word in words]
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist, squareform
        matrix = np.asarray([profiles[word] for word in words], dtype=float)
        norms = np.linalg.norm(matrix, axis=1)
        zero = norms <= 1e-12
        matrix[~zero] /= norms[~zero, None]
        distances = pdist(matrix, metric="cosine")
        distances = np.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=0.0)
        if np.any(zero):
            matrix_distance = squareform(distances)
            matrix_distance[zero, :] = 1.0
            matrix_distance[:, zero] = 1.0
            np.fill_diagonal(matrix_distance, 0.0)
            distances = squareform(matrix_distance)
        tree = linkage(distances, method="average")
        labels = fcluster(tree, t=float(tau), criterion="distance")
        grouped: dict[int, list[str]] = defaultdict(list)
        for word, label in zip(words, labels):
            grouped[int(label)].append(word)
        return sorted((sorted(group) for group in grouped.values()), key=lambda group: group[0])

    @staticmethod
    def _abstract_prediction_slot(context: tuple[str, ...], labels: dict[str, str]) -> tuple[str, ...]:
        """Replace only the immediately preceding unit by its learned class.

        Follower profiles are learned for P(next_unit | word), so their class
        membership applies to the context's final position. Applying the same
        global label to older context slots would treat a word as
        interchangeable in roles for which it was never evaluated.
        """
        if not context:
            return ()
        return (*context[:-1], labels.get(context[-1], context[-1]))

    def _validation_class_choice(self) -> tuple[float | None, float]:
        """Select coherent classes on document-heldout unseen word pairs only."""
        documents = self._class_validation_documents or []
        if not documents or self.max_order < 2:
            self.class_lambdas = {}
            self.class_selection = {
                "selection_reason": "no-class fallback: document-heldout validation unavailable",
                "validation_method": "source_grouped_document_holdout_unseen_adjacent_pairs",
                "held_pairs": 0,
                "candidate_diagnostics": [],
            }
            return None, 0.0

        # Validation counts are built from complete held-out documents; the
        # fit count tables and follower profiles below never contain these
        # documents. Hash-edge holdout from the training counts rewarded
        # memorized adjacency statistics and is intentionally removed.
        validation_counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        for text in documents:
            for sentence in split_sentences(text):
                sequence = tokenize(sentence) + [BOUNDARY]
                for index, target in enumerate(sequence):
                    context = tuple(sequence[max(0, index - self.max_order + 1):index])
                    validation_counts[context][target] += 1
        train_tables = self.raw_word_context_counts
        profiles = self._profiles_from_counts(train_tables)
        train_pairs = {
            (context[-1], target)
            for context, targets in train_tables.items() if len(context) == 1
            for target, count in targets.items() if count > 0
        }
        validation_events = []
        for context, targets in validation_counts.items():
            if not context:
                continue
            for target, count in targets.items():
                pair = (context[-1], target)
                validation_events.append((context, target, count, pair not in train_pairs))
        unseen_events = [event for event in validation_events
                         if event[3] and event[0][-1] in profiles]
        if not unseen_events:
            self.class_lambdas = {}
            self.class_selection = {
                "selection_reason": "no-class fallback: no held-out unseen pairs with learned profiles",
                "validation_method": "source_grouped_document_holdout_unseen_adjacent_pairs",
                "held_pairs": 0,
                "validation_events": len(validation_events),
                "candidate_diagnostics": [],
            }
            return None, 0.0

        # Train profiles are high-dimensional and are not retained after this
        # fit. Keep candidate cluster analysis cached by tau.
        thresholds = (0.7, 0.8, 0.9, 0.95, 0.98)
        candidate_diagnostics = []
        candidates = []
        no_class_pair_losses: dict[tuple[str, str], list[float]] = defaultdict(list)

        mapped_counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        for context, targets in train_tables.items():
            mapped_context = tuple(self._map(word) for word in context)
            for target, count in targets.items():
                mapped_counts[mapped_context][self._map(target)] += count

        def make_dist(tables, cache, *, observer_weighted=False):
            def dist(context, target):
                key = (context, target)
                if key in cache:
                    return cache[key]
                if not context:
                    unigram = mapped_counts.get((), Counter())
                    total = sum(unigram.values())
                    value = (unigram.get(target, 0) + self.alpha) / (
                        total + self.alpha * len(self.vocab)
                    )
                    if observer_weighted and self.observer is not None:
                        weight = self.observer.get_weight(self.word_level_nodes[0].id)
                        value = weight * value + (1.0 - weight) / len(self.vocab)
                else:
                    base = dist(context[1:], target)
                    local = tables.get(context, Counter())
                    total = sum(local.values())
                    if not total:
                        value = base
                    else:
                        mass = total / (total + self.alpha)
                        if observer_weighted and self.observer is not None:
                            order = min(len(context), self.max_order - 1)
                            mass *= self.observer.get_weight(self.word_level_nodes[order].id)
                        value = mass * local.get(target, 0) / total + (1.0 - mass) * base
                cache[key] = value
                return value
            return dist

        def mix_probability(context, target, class_context, lam, class_dist):
            """Use class mass only to refine the exact context's backoff mass.

            Pair targets already observed after the word keep their plain
            probability. The class expert redistributes only the remaining
            probability among unseen targets, so class induction cannot erase
            a learned edge. This is the same rule used at inference.
            """
            seen_targets = set()
            if context:
                for order in range(len(context), 0, -1):
                    local = mapped_counts.get(context[-order:], Counter())
                    if local:
                        seen_targets = set(local)
                        break
            plain = no_class_dist(context, target)
            if not seen_targets:
                return plain
            key = (context, class_context, tuple(sorted(seen_targets)))
            stats = mix_cache.get(key)
            if stats is None:
                plain_seen_mass = sum(no_class_dist(context, word) for word in seen_targets)
                class_seen_mass = sum(class_dist(class_context, word) for word in seen_targets)
                stats = (plain_seen_mass, max(0.0, 1.0 - plain_seen_mass),
                         max(0.0, 1.0 - class_seen_mass))
                mix_cache[key] = stats
            plain_seen_mass, plain_unseen_mass, class_unseen_mass = stats
            if class_unseen_mass <= 1e-12:
                return plain
            reserved_unseen_mass = max(
                plain_unseen_mass, CLASS_RESERVED_UNSEEN_MASS * lam,
            )
            if target in seen_targets:
                if reserved_unseen_mass <= plain_unseen_mass or plain_seen_mass <= 0.0:
                    return plain
                return plain * (1.0 - reserved_unseen_mass) / plain_seen_mass
            base_component = (plain / plain_unseen_mass if plain_unseen_mass > 1e-12
                              else class_dist(class_context, target) / class_unseen_mass)
            class_component = class_dist(class_context, target) / class_unseen_mass
            return reserved_unseen_mass * (
                (1.0 - lam) * base_component + lam * class_component
            )

        # The plain expert must match actual inference, including Observer
        # reliability weights. The class expert remains an interpolated
        # pooled-count distribution, mixed by its own Observer weight.
        no_class_dist = make_dist(mapped_counts, {}, observer_weighted=True)
        for context, target, count, _unseen in unseen_events:
            mapped_context = tuple(self._map(word) for word in context)
            mapped_target = self._map(target)
            pair = (context[-1], target)
            no_class_pair_losses[pair].extend(
                [-math.log(max(no_class_dist(mapped_context, mapped_target), 1e-12))] * count
            )
        no_class_pair_means = np.asarray([
            float(np.mean(losses)) for losses in no_class_pair_losses.values()
        ])
        no_class_se = (float(np.std(no_class_pair_means, ddof=1) / math.sqrt(len(no_class_pair_means)))
                       if len(no_class_pair_means) > 1 else float("inf"))

        for tau in thresholds:
            raw_clusters = self._average_link_clusters(profiles, tau)
            kept_clusters = []
            for members in raw_clusters:
                if len(members) < 2 or len(members) > CLASS_MAX_MEMBERS:
                    continue
                matrix = np.asarray([profiles[word] for word in members], dtype=float)
                norms = np.linalg.norm(matrix, axis=1)
                matrix = matrix / np.maximum(norms[:, None], 1e-12)
                centroid = matrix.mean(axis=0)
                centroid_norm = float(np.linalg.norm(centroid))
                cohesion = (float(np.mean(matrix @ centroid / max(centroid_norm, 1e-12)))
                            if centroid_norm else 0.0)
                if cohesion >= CLASS_MIN_COHESION:
                    kept_clusters.append((members, cohesion))

            labels = {}
            class_members = {}
            class_cohesion = {}
            for index, (members, cohesion) in enumerate(kept_clusters):
                class_id = f"language_class_{index:06d}"
                labels.update({word: class_id for word in members})
                class_members[class_id] = members
                class_cohesion[class_id] = cohesion

            pooled: dict[tuple[str, ...], Counter] = defaultdict(Counter)
            for context, targets in train_tables.items():
                class_context = self._abstract_prediction_slot(context, labels)
                if context and class_context != context:
                    for target, count in targets.items():
                        pooled[class_context][self._map(target)] += count
            class_dist = make_dist(pooled, {})
            mix_cache = {}

            class_lambdas = {}
            class_diagnostics = []
            for class_id, members in class_members.items():
                events = [event for event in unseen_events if labels.get(event[0][-1]) == class_id]
                pairs = sorted({(event[0][-1], event[1]) for event in events})
                if len(pairs) < CLASS_MIN_VALIDATION_PAIRS:
                    # An explicit zero distinguishes a rejected/unvalidated
                    # class from a legacy class whose weight should inherit
                    # the global compatibility default.
                    class_lambdas[class_id] = 0.0
                    class_diagnostics.append({
                        "class_id": class_id, "members": len(members),
                        "example_members": members[:12], "cohesion": class_cohesion[class_id],
                        "validation_unseen_pairs": len(pairs), "selected_weight": 0.0,
                        "selection_reason": "insufficient held-out unseen-pair evidence",
                    })
                    continue
                per_lambda: dict[float, dict[tuple[str, str], list[float]]] = {
                    lam: defaultdict(list) for lam in (0.0, *CLASS_LAMBDA_GRID)
                }
                per_seen_lambda: dict[float, dict[tuple[str, str], list[float]]] = {
                    lam: defaultdict(list) for lam in (0.0, *CLASS_LAMBDA_GRID)
                }
                for raw_context, raw_target, count, _ in events:
                    context = tuple(self._map(word) for word in raw_context)
                    target = self._map(raw_target)
                    plain = no_class_dist(context, target)
                    abstract = self._abstract_prediction_slot(raw_context, labels)
                    class_probability = class_dist(abstract, target)
                    pair = (raw_context[-1], raw_target)
                    for lam, pair_losses in per_lambda.items():
                        mixture = mix_probability(context, target, abstract, lam, class_dist)
                        pair_losses[pair].extend(
                            [-math.log(max(mixture, 1e-12))] * count
                        )
                seen_events = [
                    event for event in validation_events
                    if not event[3] and event[0]
                    and labels.get(event[0][-1]) == class_id
                ]
                for raw_context, raw_target, count, _ in seen_events:
                    context = tuple(self._map(word) for word in raw_context)
                    target = self._map(raw_target)
                    plain = no_class_dist(context, target)
                    abstract = self._abstract_prediction_slot(raw_context, labels)
                    class_probability = class_dist(abstract, target)
                    pair = (raw_context[-1], raw_target)
                    for lam, pair_losses in per_seen_lambda.items():
                        mixture = mix_probability(context, target, abstract, lam, class_dist)
                        pair_losses[pair].extend(
                            [-math.log(max(mixture, 1e-12))] * count
                        )
                means = {
                    lam: np.asarray([float(np.mean(losses)) for losses in pair_losses.values()])
                    for lam, pair_losses in per_lambda.items()
                }
                losses = {lam: float(values.mean()) for lam, values in means.items()}
                seen_means = {
                    lam: np.asarray([float(np.mean(values)) for values in pair_losses.values()])
                    for lam, pair_losses in per_seen_lambda.items()
                }
                plain_seen_loss = float(seen_means[0.0].mean()) if len(seen_means[0.0]) else 0.0
                seen_deltas_by_lambda = {
                    lam: (float(values.mean()) - plain_seen_loss if len(values) else 0.0)
                    for lam, values in seen_means.items()
                }
                eligible_lambdas = [
                    lam for lam in losses
                    if seen_deltas_by_lambda[lam] <= CLASS_MAX_SEEN_REGRESSION_NATS
                ]
                best_lam = min(eligible_lambdas, key=losses.get) if eligible_lambdas else 0.0
                best_values = means[best_lam]
                best_se = (float(np.std(best_values, ddof=1) / math.sqrt(len(best_values)))
                           if len(best_values) > 1 else 0.0)
                if losses[0.0] - losses[best_lam] <= best_se:
                    best_lam = 0.0
                else:
                    # One-standard-error selection favors the lightest mix
                    # whose held-out risk is statistically indistinguishable
                    # from the minimum.
                    cutoff = losses[best_lam] + best_se
                    best_lam = min(
                        lam for lam in eligible_lambdas if losses[lam] <= cutoff
                    )
                if len(pairs) == 1:
                    # A single held-out edge cannot estimate class-level
                    # variance. Keep its influence deliberately small; the
                    # aggregate edge-level gate below still has to establish
                    # a reliable gain for the complete clustering.
                    best_lam = min(best_lam, 0.5)
                class_lambdas[class_id] = best_lam
                class_diagnostics.append({
                    "class_id": class_id, "members": len(members),
                    "example_members": members[:12], "cohesion": class_cohesion[class_id],
                    "validation_unseen_pairs": len(pairs),
                    "validation_events": sum(event[2] for event in events),
                    "selected_weight": best_lam,
                    "validation_seen_pairs": len(seen_means[0.0]),
                    "validation_seen_pair_loss_delta_nats": seen_deltas_by_lambda[best_lam],
                    "validation_gain_nats": losses[0.0] - losses[best_lam],
                    "paired_edge_standard_error_nats": best_se if len(pairs) > 1 else None,
                    "selection_reason": "held-out unseen-pair improvement" if best_lam else
                        "no reliable held-out unseen-pair improvement",
                })

            # A class can improve its withheld combinations while blurring
            # already learned pairs. Let each class earn its own seen-pair
            # safety decision before judging the remaining classes globally.
            seen_events_by_class: dict[str, list[tuple[float, int]]] = defaultdict(list)
            for raw_context, raw_target, count, is_unseen in validation_events:
                if is_unseen or not raw_context:
                    continue
                class_id = labels.get(raw_context[-1])
                lam = class_lambdas.get(class_id, 0.0)
                if not lam:
                    continue
                context = tuple(self._map(word) for word in raw_context)
                target = self._map(raw_target)
                plain = no_class_dist(context, target)
                abstract = self._abstract_prediction_slot(raw_context, labels)
                class_probability = class_dist(abstract, target)
                mixture = mix_probability(context, target, abstract, lam, class_dist)
                loss_delta = math.log(max(plain, 1e-12) / max(mixture, 1e-12))
                seen_events_by_class[class_id].append((loss_delta, count))
            for diagnostic in class_diagnostics:
                class_id = diagnostic["class_id"]
                seen_rows = seen_events_by_class.get(class_id, [])
                if not seen_rows:
                    diagnostic["validation_seen_pairs"] = 0
                    continue
                seen_delta = sum(delta * count for delta, count in seen_rows) / sum(
                    count for _delta, count in seen_rows
                )
                diagnostic["validation_seen_pairs"] = len(seen_rows)
                diagnostic["validation_seen_pair_loss_delta_nats"] = seen_delta
                if (class_lambdas.get(class_id, 0.0) > 0.0
                        and seen_delta > CLASS_MAX_SEEN_REGRESSION_NATS):
                    class_lambdas[class_id] = 0.0
                    diagnostic["selected_weight"] = 0.0
                    diagnostic["selection_reason"] = "class harms held-out seen pairs"

            # Estimate aggregate unseen-pair gain and seen-pair harm for the
            # candidate partition. A class is active only with its own
            # evidence; global tau is chosen on pooled held-out unseen pairs.
            unseen_deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
            seen_deltas: list[float] = []
            for raw_context, raw_target, count, is_unseen in validation_events:
                if not raw_context:
                    continue
                class_id = labels.get(raw_context[-1])
                lam = class_lambdas.get(class_id, 0.0)
                if not lam:
                    continue
                context = tuple(self._map(word) for word in raw_context)
                target = self._map(raw_target)
                plain = no_class_dist(context, target)
                abstract = self._abstract_prediction_slot(raw_context, labels)
                mixture = mix_probability(context, target, abstract, lam, class_dist)
                delta = math.log(max(plain, 1e-12) / max(mixture, 1e-12))
                if is_unseen:
                    # Improvement is positive when the class mixture assigns
                    # greater likelihood to the held-out target.
                    unseen_deltas[(raw_context[-1], raw_target)].extend([-delta] * count)
                else:
                    # Loss delta is mixture minus plain, so positive values
                    # indicate seen-pair harm and trigger the safety bound.
                    seen_deltas.extend([delta] * count)
            edge_gains = np.asarray([float(np.mean(values)) for values in unseen_deltas.values()])
            gain = float(edge_gains.mean()) if len(edge_gains) else 0.0
            gain_se = (float(np.std(edge_gains, ddof=1) / math.sqrt(len(edge_gains)))
                       if len(edge_gains) > 1 else float("inf"))
            seen_delta = float(np.mean(seen_deltas)) if seen_deltas else 0.0
            rejected_for_seen_harm = seen_delta > CLASS_MAX_SEEN_REGRESSION_NATS
            if rejected_for_seen_harm or gain <= gain_se:
                class_lambdas = {}
            candidates.append({"tau": tau, "gain": gain, "gain_se": gain_se,
                               "seen_pair_loss_delta_nats": seen_delta,
                               "class_lambdas": class_lambdas,
                               "class_diagnostics": class_diagnostics,
                               "class_count": sum(value > 0.0
                                                  for value in class_lambdas.values())})
            candidate_diagnostics.append({
                "tau": tau, "unseen_pair_gain_nats": gain,
                "paired_edge_standard_error_nats": gain_se,
                "seen_pair_loss_delta_nats": seen_delta,
                "active_class_count": sum(value > 0.0 for value in class_lambdas.values()),
                "rejected_for_seen_pair_harm": rejected_for_seen_harm,
                "rejection_reason": (
                    "seen-pair loss exceeded safety bound" if rejected_for_seen_harm else
                    "unseen-pair gain did not exceed its paired standard error"
                    if gain <= gain_se else None
                ),
                "class_quality": class_diagnostics,
            })
            if (rejected_for_seen_harm or gain <= gain_se or not class_lambdas):
                candidates.pop()

        if candidates:
            # Each active class has already earned a per-class held-out gain,
            # and the complete candidate has passed the paired-edge and
            # seen-pair gates. Prefer the validated partition that retains the
            # most independently useful class nodes; aggregate gain breaks
            # ties between equally expressive candidates.
            selected = max(candidates, key=lambda row: (
                row["class_count"], row["gain"], -float(row["tau"]),
            ))
        else:
            selected = {"tau": None, "gain": 0.0, "gain_se": no_class_se,
                        "seen_pair_loss_delta_nats": 0.0,
                        "class_lambdas": {}, "class_diagnostics": [], "class_count": 0}
        if selected["gain"] <= selected["gain_se"] or not selected["class_lambdas"]:
            self.class_lambdas = {}
            self.class_selection = {
                "validation_method": "source_grouped_document_holdout_unseen_adjacent_pairs",
                "selection_reason": "no-class fallback: no reliable held-out unseen-pair gain",
                "held_pairs": len({(event[0][-1], event[1]) for event in unseen_events}),
                "validation_events": len(validation_events),
                "unseen_validation_events": sum(event[2] for event in unseen_events),
                "unseen_pair_gain_nats": selected["gain"],
                "paired_edge_standard_error_nats": selected["gain_se"],
                "candidate_diagnostics": candidate_diagnostics,
            }
            return None, 0.0
        self.class_lambdas = dict(selected["class_lambdas"])
        self.class_lambda = max(self.class_lambdas.values(), default=0.0)
        self.class_selection = {
            "validation_method": "source_grouped_document_holdout_unseen_adjacent_pairs",
            "selection_reason": "per-class held-out unseen-pair reinforcement",
            "held_pairs": len({(event[0][-1], event[1]) for event in unseen_events}),
            "validation_events": len(validation_events),
            "unseen_validation_events": sum(event[2] for event in unseen_events),
            "selected_tau": selected["tau"],
            "selected_lambda_by_class": self.class_lambdas,
            "selected_validation_gain_nats": selected["gain"],
            "paired_edge_standard_error_nats": selected["gain_se"],
            "selected_seen_pair_loss_delta_nats": selected["seen_pair_loss_delta_nats"],
            "class_quality": selected["class_diagnostics"],
            "candidate_diagnostics": candidate_diagnostics,
        }
        return selected["tau"], max(self.class_lambdas.values(), default=0.0)

    def _fit_class_nodes_from_counts(self, *, refit: bool) -> None:
        """Induce/freeze classes, refresh compact projections, and pool counts."""
        profiles = self._profiles_from_counts(self.raw_word_context_counts)
        if refit:
            self.class_tau, self.class_lambda = self._validation_class_choice()
            profiles = self._profiles_from_counts(self.raw_word_context_counts)
            clusters = self._average_link_clusters(profiles, self.class_tau)
            self.class_labels = {}
            valid_clusters = []
            for members in clusters:
                if len(members) < 2 or len(members) > CLASS_MAX_MEMBERS:
                    continue
                matrix = np.asarray([profiles[word] for word in members], dtype=float)
                norms = np.linalg.norm(matrix, axis=1)
                matrix = matrix / np.maximum(norms[:, None], 1e-12)
                centroid = matrix.mean(axis=0)
                centroid_norm = float(np.linalg.norm(centroid))
                cohesion = (float(np.mean(matrix @ centroid / max(centroid_norm, 1e-12)))
                            if centroid_norm else 0.0)
                if cohesion >= CLASS_MIN_COHESION:
                    valid_clusters.append(members)
            for index, members in enumerate(valid_clusters):
                class_id = f"language_class_{index:06d}"
                if class_id not in self.class_lambdas and self.class_lambda > 0:
                    # Compatibility for callers that inject a class-choice
                    # policy. Normal training always records per-class values.
                    self.class_lambdas[class_id] = self.class_lambda
                if self.class_lambdas.get(class_id, 0.0) <= 0:
                    continue
                self.class_labels.update({word: class_id for word in members})
            self._class_membership_frozen = True
        # The full follower vectors are transient clustering inputs. Persisting
        # one mean and variance vector per class duplicates thousands of
        # dimensions even though HFN nodes use only the stable four-dimensional
        # projection. Recompute that compact summary after load and partial_fit.
        self.class_profile_features = list(self._current_profile_features)
        self.class_profiles = {}
        self.class_profile_variances = {}
        for class_id in sorted(set(self.class_labels.values())):
            members = sorted(word for word, label in self.class_labels.items() if label == class_id)
            projected = np.asarray([
                self._project_follower_profile(
                    profiles.get(word, np.zeros(len(self.class_profile_features))),
                    self.class_profile_features,
                ) for word in members
            ])
            self.class_profiles[class_id] = projected.mean(axis=0)
            self.class_profile_variances[class_id] = projected.var(axis=0) + 0.05 ** 2
        self.class_context_counts = defaultdict(Counter)
        if not self.class_labels or not any(self.class_lambdas.values()):
            return
        for raw_context, targets in self.raw_word_context_counts.items():
            if not raw_context:
                continue
            class_context = self._abstract_prediction_slot(raw_context, self.class_labels)
            if class_context == raw_context:
                continue
            for target, count in targets.items():
                self.class_context_counts[class_context][self._map(target)] += count

    def _refit_selected_classes_on_full_training_counts(self) -> dict:
        """Refit selected class profiles on all initial-fit documents.

        Selection and lambda tuning use document-heldout pairs. After that
        choice is frozen, the reserved fit documents return to the training
        counts and class profiles are recomputed on the complete training
        partition, as in ordinary train/validation refit. Active weights are
        transferred by member overlap; classes with no earned weight stay at
        zero. Subsequent ``partial_fit`` calls still preserve membership.
        """
        prior_labels = dict(self.class_labels)
        prior_weights = dict(self.class_lambdas)
        profiles = self._profiles_from_counts(self.raw_word_context_counts)
        clusters = self._average_link_clusters(profiles, self.class_tau)
        self.class_labels = {}
        self.class_lambdas = {}
        valid_clusters = []
        for members in clusters:
            if len(members) < 2 or len(members) > CLASS_MAX_MEMBERS:
                continue
            matrix = np.asarray([profiles[word] for word in members], dtype=float)
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            centroid = matrix.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            cohesion = float(np.mean(matrix @ centroid / max(norm, 1e-12))) if norm else 0.0
            if cohesion >= CLASS_MIN_COHESION:
                valid_clusters.append(members)
        for index, members in enumerate(valid_clusters):
            class_id = f"language_class_{index:06d}"
            inherited = [prior_weights.get(prior_labels[word], 0.0)
                         for word in members if word in prior_labels]
            weight = sum(inherited) / len(members) if inherited else 0.0
            self.class_lambdas[class_id] = weight
            if weight > 0.0:
                self.class_labels.update({word: class_id for word in members})
        self.class_lambda = max(self.class_lambdas.values(), default=0.0)
        self._class_membership_frozen = True
        self.class_profile_features = list(self._current_profile_features)
        self.class_profiles = {}
        self.class_profile_variances = {}
        for class_id in sorted(set(self.class_labels.values())):
            members = sorted(word for word, label in self.class_labels.items()
                             if label == class_id)
            projected = np.asarray([
                self._project_follower_profile(
                    profiles.get(word, np.zeros(len(self.class_profile_features))),
                    self.class_profile_features,
                ) for word in members
            ])
            self.class_profiles[class_id] = projected.mean(axis=0)
            self.class_profile_variances[class_id] = projected.var(axis=0) + 0.05 ** 2
        self.class_context_counts = defaultdict(Counter)
        for raw_context, targets in self.raw_word_context_counts.items():
            if not raw_context:
                continue
            class_context = self._abstract_prediction_slot(raw_context, self.class_labels)
            if class_context == raw_context:
                continue
            for target, count in targets.items():
                self.class_context_counts[class_context][self._map(target)] += count
        return {
            "method": "full_training_profile_refit_after_heldout_selection",
            "classes": len(valid_clusters),
            "active_classes": len(set(self.class_labels.values())),
            "members": len(self.class_labels),
            "selected_tau": self.class_tau,
        }

    def _build_hfn_nodes(
        self, *, progress_callback: Callable[[dict], None] | None = None,
    ) -> None:
        def report(stage: str, **details) -> None:
            if progress_callback:
                progress_callback({"fit_stage": "hfn_node_build", "node_stage": stage,
                                   **details})

        self.nodes.clear()
        self._class_mix_probability_cache = OrderedDict()
        self._class_mix_distribution_cache = OrderedDict()
        self._class_mix_mass_cache.clear()
        self.class_node_registry.clear()
        self.char_nodes.clear()
        self.word_nodes.clear()
        self.context_nodes.clear()
        self.class_context_nodes.clear()
        self.phrase_nodes.clear()
        self.char_context_nodes.clear()
        self.sentence_nodes.clear()
        follower_profiles = (self._profiles_from_counts(self.raw_word_context_counts)
                             if self.class_nodes_enabled else {})
        for ch in self.char_vocab:
            if len(ch) != 1:
                continue
            node = HFN(np.array([float(ord(ch)) / 0x10FFFF, 0.0, 0.0, 0.0]),
                       np.full(4, 1e-6), id=self._safe_id("char_", ch), use_diag=True)
            node.metadata = {"type": "character", "value": ch}
            self.char_nodes[ch] = self.nodes[node.id] = node
        report("characters", state="complete", nodes=len(self.char_nodes))
        for word in self.vocab:
            count = self.word_counts.get(word, 0)
            profile = follower_profiles.get(word)
            if profile is not None:
                word_mu = self._project_follower_profile(profile, self._current_profile_features)
                word_sigma = np.full(4, 0.05 ** 2)
            else:
                word_mu = np.array([float(len(word)), math.log1p(count), 0.0, 0.0])
                word_sigma = np.array([1e-6, 1.0 / (count + 1), 1e-6, 1e-6])
            node = HFN(word_mu, word_sigma,
                       id=self._safe_id("word_", word), use_diag=True)
            node.metadata = {"type": "word", "value": word}
            if profile is not None:
                node.metadata["follower_profile_dimensions"] = int(profile.size)
                node.metadata["follower_profile_nonzero_features"] = int(np.count_nonzero(profile))
            if word not in (BOUNDARY, UNK):
                for char_node in (self.char_nodes[c] for c in word if c in self.char_nodes):
                    node.add_child(char_node, relation="character_sequence")
            self.word_nodes[word] = self.nodes[node.id] = node
        report("words", state="complete", nodes=len(self.word_nodes))
        for class_id in self.class_profiles:
            members = sorted(word for word, label in self.class_labels.items() if label == class_id)
            if len(members) < 2:
                continue
            member_nodes = [self.word_nodes[word] for word in members if word in self.word_nodes]
            if len(member_nodes) < 2:
                continue
            member_profiles = np.asarray([member.mu for member in member_nodes])
            mu = member_profiles.mean(axis=0)
            profile_variance = member_profiles.var(axis=0)
            sigma = profile_variance + 0.05 ** 2
            # Refresh the persisted summaries from the actual HFN members. This
            # also handles members omitted from the current mapped vocabulary.
            self.class_profiles[class_id] = mu.copy()
            self.class_profile_variances[class_id] = sigma.copy()
            node = HFN(mu, sigma, id=class_id, use_diag=True)
            node.metadata = {"type": "word_class", "members": members,
                             "follower_profile_dimensions": len(self.class_profile_features),
                             "follower_profile_feature_count": len(self.class_profile_features),
                             "initial_observer_weight": self.class_lambdas.get(class_id, 0.0),
                             "cohesion": next((row.get("cohesion") for row in
                                               (self.class_selection or {}).get("class_quality", [])
                                               if row.get("class_id") == class_id), None),
                             "projected_profile_mean": mu.tolist(),
                             "projected_profile_variance": profile_variance.tolist()}
            for word in members:
                if word in self.word_nodes:
                    node.add_child(self.word_nodes[word], relation="member")
            node.relation_type = "distributional_substitution_class"
            node.relation_params = {"profile": "idf_weighted_positive_pmi_of_followers",
                                   "cluster_threshold": self.class_tau,
                                   "substitution_slot": "immediately_preceding_unit",
                                   "member_count": len(member_nodes)}
            node.add_relation(inputs=member_nodes)
            self.class_node_registry[class_id] = self.nodes[class_id] = node
        report("classes", state="complete", nodes=len(self.class_node_registry))
        for context, targets in self.char_counts.items():
            total = sum(targets.values())
            entropy = -sum((n / total) * math.log(n / total) for n in targets.values()) if total else 0.0
            categorical_variance = 1.0 - sum((n / total) ** 2 for n in targets.values()) if total else 1.0
            node = HFN(np.array([entropy, math.log1p(total), 0.0, 0.0]),
                       np.array([max(categorical_variance, 1e-9), 1.0 / (total + 1.0), 1e-6, 1e-6]),
                       id=self._safe_id("char_context_", "\x1f".join(context)), use_diag=True)
            node.metadata = {"type": "next_character_distribution", "context": list(context),
                             "observations": total, "target_counts": dict(targets),
                             "sigma_dimensions": ["categorical_gini_variance", "inverse_n_plus_one"]}
            for char in context:
                if char in self.char_nodes:
                    node.add_child(self.char_nodes[char], relation="character_context")
            self.char_context_nodes[context] = self.nodes[node.id] = node
        report("character_contexts", state="complete", nodes=len(self.char_context_nodes))
        for context, targets in self.counts.items():
            total = sum(targets.values())
            entropy = -sum((n / total) * math.log(n / total) for n in targets.values()) if total else 0.0
            categorical_variance = 1.0 - sum((n / total) ** 2 for n in targets.values()) if total else 1.0
            # mu/sigma summarize learned predictability; sparse target counts
            # live in metadata so storage does not grow as O(vocabulary*contexts).
            node = HFN(np.array([entropy, math.log1p(total), 0.0, 0.0]),
                       np.array([max(categorical_variance, 1e-9), 1.0 / (total + 1.0), 1e-6, 1e-6]),
                       id=self._safe_id("context_", "\x1f".join(context)), use_diag=True)
            node.metadata = {"type": "next_unit_distribution", "context": list(context), "observations": total,
                             "target_counts": dict(targets),
                             "sigma_dimensions": ["categorical_gini_variance", "inverse_n_plus_one"]}
            for word in context:
                if word in self.word_nodes:
                    node.add_child(self.word_nodes[word], relation="phrase_context")
            self.context_nodes[context] = self.nodes[node.id] = node
            if context:
                self.phrase_nodes[context] = node
        report("word_contexts", state="complete", nodes=len(self.context_nodes))
        # A pooled class context is itself a predictive HFN pattern. It links
        # the abstract class nodes and any exact word nodes in its slot, and
        # stores pooled follower sufficient statistics on the node. The
        # dictionary is only a fast index for these graph patterns.
        for context, targets in self.class_context_counts.items():
            if not context:
                continue
            total = sum(targets.values())
            entropy = (-sum((count / total) * math.log(count / total)
                            for count in targets.values()) if total else 0.0)
            categorical_variance = (1.0 - sum((count / total) ** 2
                                               for count in targets.values()) if total else 1.0)
            node = HFN(np.array([entropy, math.log1p(total), 0.0, 0.0]),
                       np.array([max(categorical_variance, 1e-9),
                                 1.0 / (total + 1.0), 1e-6, 1e-6]),
                       id=self._safe_id("class_context_", "\x1f".join(context)), use_diag=True)
            node.metadata = {"type": "class_conditioned_next_unit_distribution",
                             "context": list(context), "observations": total,
                             "target_counts": dict(targets),
                             "sigma_dimensions": ["categorical_gini_variance", "inverse_n_plus_one"]}
            context_nodes = []
            for token in context:
                child = self.class_node_registry.get(token) or self.word_nodes.get(token)
                if child is not None:
                    node.add_child(child)
                    context_nodes.append(child)
            node.relation_type = "next_unit_given_class_context"
            node.relation_params = {"context_order": len(context),
                                   "context_slots": list(range(len(context))),
                                   "abstracted_slots": [index for index, token in enumerate(context)
                                                        if token in self.class_node_registry]}
            node.add_relation(
                inputs=context_nodes,
                outputs=[self.word_nodes[target] for target in targets if target in self.word_nodes],
            )
            self.class_context_nodes[context] = self.nodes[node.id] = node
        report("class_contexts", state="complete", nodes=len(self.class_context_nodes))
        for sentence, count in self.sentence_counts.items():
            words = tokenize(sentence)
            node = HFN(np.array([float(len(words)), math.log1p(count), 0.0, 0.0]),
                       np.array([1.0 / (count + 1.0), 1.0 / (len(words) + 1.0), 1e-6, 1e-6]),
                       id=self._safe_id("sentence_", sentence), use_diag=True)
            node.metadata = {"type": "sentence", "value": sentence, "word_sequence": words,
                             "observations": count}
            seen_contexts = set()
            for end in range(len(words) + 1):
                for order in range(1, min(self.max_order - 1, end) + 1):
                    context = tuple(self._map(w) for w in words[end-order:end])
                    phrase = self.phrase_nodes.get(context)
                    if phrase is not None and phrase.id not in seen_contexts:
                        node.add_child(phrase, relation="sentence_phrase")
                        seen_contexts.add(phrase.id)
            self.sentence_nodes[sentence] = self.nodes[node.id] = node
        report("sentences", state="complete", nodes=len(self.sentence_nodes),
               total_nodes=len(self.nodes))

    def _initialize_pattern_observer(
        self, *, progress_callback: Callable[[dict], None] | None = None,
    ) -> None:
        """Create HFN Observer level nodes over the registered language graph.

        Compact fixed-width level nodes carry Observer state. The full
        fixed-width character-to-sentence graph remains registered in the same
        HFN Forest and each level node parents its detailed patterns.
        """
        self.pattern_forest = Forest(D=4, forest_id="language_pattern_hierarchy")
        self.word_level_nodes = {}
        self.char_level_nodes = {}
        levels = []
        for order in range(self.max_order):
            for level, registry in (("word", self.word_level_nodes), ("character", self.char_level_nodes)):
                node = HFN(np.array([float(order), float(order > 0), 0.0, 0.0]),
                           np.ones(4), id=f"language_{level}_order_{order}", use_diag=True)
                node.metadata = {"type": "language_prediction_level", "unit": level,
                                 "context_order": order}
                registry[order] = node
                levels.append(node)
        # One full HFN Gaussian reduction keeps large corpus restoration and
        # fitting bounded; separate registrations would each rescan the graph.
        registered_nodes = [*self.nodes.values(), *levels]
        if progress_callback:
            progress_callback({"fit_stage": "observer_initialization",
                               "observer_stage": "registering_patterns", "state": "running",
                               "hfn_nodes": len(self.nodes)})
        if hasattr(self.pattern_forest, "register_many"):
            self.pattern_forest.register_many(registered_nodes)
        else:  # Compatibility with the pre-batched Forest shipped in clean releases.
            for node in registered_nodes:
                if node.mu.shape != (self.pattern_forest._D,):
                    raise ValueError(f"node {node.id!r} is not compatible with the language Forest")
                self.pattern_forest._registry.setdefault(node.id, node)
            self.pattern_forest.rebuild_index()
        if progress_callback:
            progress_callback({"fit_stage": "observer_initialization",
                               "observer_stage": "patterns_registered", "state": "complete",
                               "forest_nodes": len(self.pattern_forest)})
        self.observer = Observer(self.pattern_forest, alpha_gain=0.1, w_init=0.5,
                                 beta_loss=0.0, residual_surprise_threshold=float("inf"))
        for node in levels:
            self.observer.register(node, initial_weight=0.5)
        for class_id, node in self.class_node_registry.items():
            self.observer.register(
                node, initial_weight=self.class_lambdas.get(class_id, self.class_lambda)
            )
        for word, node in self.word_nodes.items():
            if word not in (BOUNDARY, UNK):
                self.word_level_nodes[0].add_child(node, relation="weights_word_pattern")
        for context, node in self.context_nodes.items():
            self.word_level_nodes[min(len(context), self.max_order - 1)].add_child(
                node, relation="weights_phrase_pattern")
        for context, node in self.class_context_nodes.items():
            self.word_level_nodes[min(len(context), self.max_order - 1)].add_child(
                node, relation="weights_class_phrase_pattern")
        for char, node in self.char_nodes.items():
            self.char_level_nodes[0].add_child(node, relation="weights_character_pattern")
        for context, node in self.char_context_nodes.items():
            self.char_level_nodes[min(len(context), self.max_order - 1)].add_child(
                node, relation="weights_character_context")
        for node in self.sentence_nodes.values():
            self.word_level_nodes[self.max_order - 1].add_child(node, relation="weights_sentence_pattern")

    def _reinforce_class_observer_weights(self, documents: Iterable[str]) -> dict:
        """Prequentially reinforce classes on heldout unseen adjacent pairs.

        Each target is scored before its class weight changes. The class expert
        receives positive reinforcement when its likelihood beats the exact
        word-context expert and a penalty otherwise. Validation documents are
        never added to the count tables until this pass has completed.
        """
        if not self.class_node_registry or self.observer is None:
            return {"events": 0, "updated_classes": 0, "weights": {}}
        train_pairs = {
            (context[-1], target)
            for context, targets in self.raw_word_context_counts.items()
            if len(context) == 1 for target, count in targets.items() if count > 0
        }
        before = {class_id: self.observer.get_weight(class_id)
                  for class_id in self.class_node_registry}
        events = 0
        for text in documents:
            for sentence in split_sentences(text):
                sequence = tokenize(sentence) + [BOUNDARY]
                for index, raw_target in enumerate(sequence):
                    raw_context = tuple(sequence[max(0, index - self.max_order + 1):index])
                    if not raw_context:
                        continue
                    class_id = self.class_labels.get(raw_context[-1])
                    if class_id not in self.class_node_registry:
                        continue
                    if (raw_context[-1], raw_target) in train_pairs:
                        continue
                    mapped_context = tuple(self._map(word) for word in raw_context)
                    target = self._map(raw_target)
                    abstract = self._abstract_prediction_slot(raw_context, self.class_labels)
                    class_context = tuple(
                        token if token == class_id else self._map(token) for token in abstract
                    )
                    plain_probability = self._weighted_probability(
                        mapped_context, target, apply_class_mix=False
                    )
                    class_probability = self._class_probability_for_context(class_context, target)
                    if class_probability > plain_probability:
                        self.observer.boost_id(class_id, gain=0.1)
                    else:
                        self.observer.penalize_id(class_id, penalty=0.02)
                    events += 1
        weights = {class_id: self.observer.get_weight(class_id)
                   for class_id in self.class_node_registry}
        dropped = []
        for class_id, weight in list(weights.items()):
            if weight < 0.005:
                self.observer.set_weight(class_id, 0.0)
                weights[class_id] = 0.0
                dropped.append(class_id)
            self.class_lambdas[class_id] = weights[class_id]
        self.class_lambda = max(self.class_lambdas.values(), default=0.0)
        return {
            "events": events,
            "updated_classes": sum(abs(weights[key] - before[key]) > 1e-12 for key in weights),
            "weights_before": before,
            "weights": weights,
            "dropped_classes": dropped,
        }

    def _fit_pattern_observer(
        self, *, progress_callback: Callable[[dict], None] | None = None,
    ) -> None:
        self._initialize_pattern_observer(progress_callback=progress_callback)
        self._set_pattern_observer_reliability()

    def _set_pattern_observer_reliability(self) -> None:
        """Update level weights on the existing graph without rebuilding it."""
        if self.observer is None:
            raise RuntimeError("pattern Observer must be initialized before setting reliability")
        for registry, counts_table in ((self.word_level_nodes, self.counts),
                                        (self.char_level_nodes, self.char_counts)):
            outcomes = [[0, 0] for _ in range(self.max_order)]
            for context, counts in counts_table.items():
                order = len(context)
                if order >= self.max_order:
                    continue
                # Rank targets once per context. Sorting inside the target
                # loop made this O(V^2 log V) for a context with V followers,
                # which could stall incremental Observer refresh on large
                # vocabularies.
                ordered_targets = sorted(
                    counts.items(), key=lambda item: (item[1], item[0]), reverse=True,
                )
                for target, frequency in counts.items():
                    outcomes[order][1] += frequency
                    if self._leave_one_out_correct(counts, target, ordered=ordered_targets):
                        outcomes[order][0] += frequency
            for order, (correct, total) in enumerate(outcomes):
                # Smoothed empirical reliability keeps unseen levels usable.
                reliability = (correct + 1.0) / (total + 2.0)
                self.observer.set_weight(registry[order].id, reliability)

    @staticmethod
    def _leave_one_out_correct(
        counts: Counter[str], target: str, *, ordered: list[tuple[str, int]] | None = None,
    ) -> bool:
        """Whether the modal prediction remains correct after removing this event."""
        if not counts:
            return False
        if ordered is None:
            ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
        winner, winner_count = ordered[0]
        if winner != target:
            return False
        remaining = winner_count - 1
        if len(ordered) == 1:
            return remaining > 0
        runner, runner_count = ordered[1]
        if remaining > runner_count:
            return True
        if remaining < runner_count:
            return False
        return target > runner

    def train_observer_weights(
        self, training_documents: Iterable[str], *, character: bool = False,
        passes: int = 3, max_events: int = 20_000,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict:
        """Fit HFN level weights on training-only leave-one-event-out loss.

        This is the repeatable parameter-learning step for the count model:
        counts remain fixed while coordinate-descent passes update Observer
        weights against a supervised next-unit objective. Each training event
        is scored after subtracting that event from its unigram and suffix
        context counts, avoiding direct self-prediction. Validation calibration
        should still follow, and the test set must remain untouched.
        """
        if passes < 1 or max_events < 1:
            raise ValueError("passes and max_events must be positive")
        documents = list(training_documents)
        registry = self.char_level_nodes if character else self.word_level_nodes
        counts_table = self.char_counts if character else self.counts
        vocabulary = self.char_vocab if character else self.vocab
        alpha = self.char_alpha if character else self.alpha
        if not registry or self.observer is None:
            raise RuntimeError("pattern Observer is not initialized")

        events: Counter[tuple[tuple[str, ...], str]] = Counter()
        for text in documents:
            for sentence in split_sentences(text):
                if character:
                    sequence = ([BOUNDARY] * (self.max_order - 1)
                                + list(sentence.lower()) + [BOUNDARY])
                    for index in range(self.max_order - 1, len(sequence)):
                        raw_target = sequence[index]
                        target = raw_target if raw_target in self.char_vocab else UNK
                        context = tuple(sequence[index - self.max_order + 1:index]) if self.max_order > 1 else ()
                        context = tuple(char if char in self.char_vocab else UNK for char in context)
                        events[(context, target)] += 1
                else:
                    sequence = [self._map(word) for word in tokenize(sentence)] + [BOUNDARY]
                    for index, target in enumerate(sequence):
                        context = tuple(sequence[max(0, index - self.max_order + 1):index])
                        events[(context, target)] += 1
        total_events = sum(events.values())
        if not total_events:
            raise ValueError("training documents contain no next-unit events")

        ordered_events = sorted(
            events.items(),
            key=lambda item: hashlib.sha256(repr(item[0]).encode("utf-8")).digest(),
        )
        stride = max(1, math.ceil(total_events / max_events))

        def selected_count(start: int, frequency: int) -> int:
            first = start + (-start % stride)
            if first >= start + frequency:
                return 0
            return 1 + (start + frequency - 1 - first) // stride

        optimization_events = 0
        selected_events = []
        event_offset = 0
        for (context, target), frequency in ordered_events:
            used = selected_count(event_offset, frequency)
            optimization_events += used
            if used:
                selected_events.append((context, target, used))
            event_offset += frequency

        # Materialize each selected event's fixed count statistics once. The
        # previous coordinate-descent loop rescanned every unique context and
        # re-summed its unigram and suffix counters for every candidate weight;
        # on large new-source rounds that made the 20k-event cap misleading.
        unigram = counts_table.get((), Counter())
        unigram_total = sum(unigram.values())
        vocabulary_size = len(vocabulary)
        scored_events = []
        for context, target, used in selected_events:
            target_unigram = max(0, unigram.get(target, 0) - 1)
            remaining_unigrams = max(0, unigram_total - 1)
            unigram_probability = (target_unigram + alpha) / (
                remaining_unigrams + alpha * vocabulary_size
            )
            suffix_probabilities = []
            for order in range(1, min(len(context), self.max_order - 1) + 1):
                local = counts_table.get(context[-order:], Counter())
                remaining_total = sum(local.values()) - 1
                if remaining_total <= 0:
                    continue
                remaining_target = max(0, local.get(target, 0) - 1)
                evidence_mass = remaining_total / (remaining_total + alpha)
                suffix_probabilities.append((order, evidence_mass,
                                             remaining_target / remaining_total))
            scored_events.append((used, unigram_probability, suffix_probabilities))

        def loss_for(weights: list[float]) -> float:
            nll = observed = 0
            for used, unigram_probability, suffix_probabilities in scored_events:
                probability = unigram_probability
                probability = (weights[0] * probability
                               + (1.0 - weights[0]) / vocabulary_size)
                for order, evidence_mass, conditional_probability in suffix_probabilities:
                    local_mass = weights[order] * evidence_mass
                    probability = (local_mass * conditional_probability
                                   + (1.0 - local_mass) * probability)
                nll -= used * math.log(max(probability, 1e-15))
                observed += used
            return nll / max(observed, 1)

        weights = [self.observer.get_weight(registry[order].id)
                   for order in range(self.max_order)]
        initial = list(weights)
        before = loss_for(weights)
        candidates = (0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)
        completed = 0
        passes_run = 0
        for pass_index in range(passes):
            changed = False
            for order in range(self.max_order):
                current = weights[order]
                best_weight, best_loss = current, loss_for(weights)
                for candidate in sorted(set(candidates) | {current}):
                    trial = list(weights)
                    trial[order] = candidate
                    loss = loss_for(trial)
                    completed += 1
                    if progress_callback:
                        progress_callback({"unit": "character" if character else "word",
                                           "pass": pass_index + 1, "order": order,
                                           "candidate_weight": candidate,
                                           "training_loo_cross_entropy": loss,
                                           "completed_candidates": completed})
                    if loss < best_loss - 1e-12:
                        best_weight, best_loss = candidate, loss
                weights[order] = best_weight
                changed |= best_weight != current
            passes_run = pass_index + 1
            if not changed:
                break
        for order, node in registry.items():
            self.observer.set_weight(node.id, weights[order])
        if not character:
            self._character_word_prior_cache = None
        after = loss_for(weights)
        return {"unit": "character" if character else "word",
                "objective": "training_leave_one_event_out_next_unit_cross_entropy",
                "training_events": total_events, "optimization_events": optimization_events,
                "max_events": max_events, "unique_event_contexts": len(ordered_events),
                "scored_event_contexts": len(scored_events),
                "stride": stride, "initial_weights": {str(i): value for i, value in enumerate(initial)},
                "selected_weights": {str(i): value for i, value in enumerate(weights)},
                "training_loo_cross_entropy_before": before,
                "training_loo_cross_entropy_after": after,
                "passes_requested": passes, "passes_run": passes_run}

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

    def _class_mixed_probability(
        self, context: tuple[str, ...], target: str, plain_probability: float,
    ) -> float | None:
        """Score one class-mixed target without materializing a vocabulary map.

        Class mixing reserves probability mass for unseen continuations while
        preserving the base predictor on observed continuations. Only the mass
        of the observed set under the plain and class experts is needed for a
        single target. Computing those sums scales with observed followers,
        rather than vocabulary size, and avoids allocating a dense class map
        for every token in a corpus evaluation.
        """
        class_context = self._abstract_prediction_slot(context, self.class_labels)
        if class_context == context or class_context not in self.class_context_nodes:
            return None
        class_id = class_context[-1]
        weight = (self.observer.get_weight(class_id)
                  if self.observer is not None and class_id in self.class_node_registry
                  else self.class_lambdas.get(class_id, self.class_lambda))
        if weight <= 0.0:
            return None

        seen_targets = set()
        for order in range(len(context), 0, -1):
            local = self.counts.get(context[-order:], Counter())
            if local:
                seen_targets = set(local)
                break
        if not seen_targets:
            return None

        order_weights = (tuple(self.observer.get_weight(node.id)
                               for _, node in sorted(self.word_level_nodes.items()))
                         if self.observer is not None else ())
        key = (context, class_context, weight, order_weights, self.alpha)
        cached = self._class_mix_mass_cache.get(key)
        if cached is None:
            plain_seen_mass = sum(
                self._weighted_probability(
                    context, word, apply_class_mix=False, apply_character_mix=False,
                ) for word in seen_targets
            )
            class_seen_mass = sum(
                self._class_probability_for_context(class_context, word)
                for word in seen_targets
            )
            cached = (plain_seen_mass, max(0.0, 1.0 - plain_seen_mass),
                      max(0.0, 1.0 - class_seen_mass))
            self._class_mix_mass_cache[key] = cached
            while len(self._class_mix_mass_cache) > CLASS_MIX_MASS_CACHE_MAX_ENTRIES:
                self._class_mix_mass_cache.popitem(last=False)
        else:
            self._class_mix_mass_cache.move_to_end(key)

        plain_seen_mass, plain_unseen_mass, class_unseen_mass = cached
        if class_unseen_mass <= 1e-12:
            return plain_probability
        reserved_unseen_mass = max(
            plain_unseen_mass, CLASS_RESERVED_UNSEEN_MASS * weight,
        )
        if target in seen_targets:
            if reserved_unseen_mass > plain_unseen_mass and plain_seen_mass > 0.0:
                return plain_probability * (1.0 - reserved_unseen_mass) / plain_seen_mass
            return plain_probability
        if reserved_unseen_mass > plain_unseen_mass and plain_seen_mass <= 0.0:
            return plain_probability
        class_probability = self._class_probability_for_context(class_context, target)
        if plain_unseen_mass > 1e-12:
            plain_component = plain_probability / plain_unseen_mass
        else:
            plain_component = class_probability / class_unseen_mass
        class_component = class_probability / class_unseen_mass
        return reserved_unseen_mass * (
            (1.0 - weight) * plain_component + weight * class_component
        )

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

    def _class_mixed_distribution(
        self, context: tuple[str, ...], *, plain_distribution: dict[str, float] | None = None,
    ) -> dict[str, float] | None:
        """Apply a class prior only over probability mass reserved for unseen edges.

        The exact context retains its learned target probabilities. The class
        node redistributes only the remaining mass across unseen targets.
        This protects observed word pairs while allowing the hierarchy to
        rank plausible but unobserved followers.
        """
        class_context = self._abstract_prediction_slot(context, self.class_labels)
        if class_context == context or class_context not in self.class_context_nodes:
            return None
        class_id = class_context[-1]
        weight = (self.observer.get_weight(class_id)
                  if self.observer is not None and class_id in self.class_node_registry
                  else self.class_lambdas.get(class_id, self.class_lambda))
        if weight <= 0.0:
            return None
        if self.observer is not None and plain_distribution is None:
            order_weights = tuple(
                self.observer.get_weight(node.id)
                for _, node in sorted(self.word_level_nodes.items())
            )
        else:
            order_weights = ()
        mode = "count" if plain_distribution is not None else "observer"
        cache_key = (mode, context, class_context, weight, order_weights, self.alpha)
        cache = getattr(self, "_class_mix_distribution_cache", None)
        if cache is None:
            cache = self._class_mix_distribution_cache = OrderedDict()
        cached = cache.get(cache_key)
        if cached is not None:
            cache.move_to_end(cache_key)
            return cached
        if plain_distribution is None:
            plain_distribution = {
                word: self._weighted_probability(
                    context, word, apply_class_mix=False, apply_character_mix=False,
                ) for word in self.vocab
            }
        seen_targets = set()
        if context:
            for order in range(len(context), 0, -1):
                local = self.counts.get(context[-order:], Counter())
                if local:
                    seen_targets = set(local)
                    break
        if not seen_targets:
            return None
        seen_mass = sum(plain_distribution.get(word, 0.0) for word in seen_targets)
        plain_unseen_mass = max(0.0, 1.0 - seen_mass)
        class_distribution = self._class_distribution_for_context(class_context)
        class_unseen_mass = max(
            0.0, 1.0 - sum(class_distribution.get(word, 0.0) for word in seen_targets)
        )
        if class_unseen_mass <= 1e-12:
            return plain_distribution
        reserved_unseen_mass = max(
            plain_unseen_mass, CLASS_RESERVED_UNSEEN_MASS * weight,
        )
        if reserved_unseen_mass > plain_unseen_mass and seen_mass <= 0.0:
            return plain_distribution
        mixed = {}
        for word in self.vocab:
            if word in seen_targets:
                mixed[word] = (
                    plain_distribution.get(word, 0.0)
                    * (1.0 - reserved_unseen_mass) / seen_mass
                    if reserved_unseen_mass > plain_unseen_mass else
                    plain_distribution.get(word, 0.0)
                )
            else:
                base_component = (
                    plain_distribution.get(word, 0.0) / plain_unseen_mass
                    if plain_unseen_mass > 1e-12 else
                    class_distribution.get(word, 0.0) / class_unseen_mass
                )
                class_component = class_distribution.get(word, 0.0) / class_unseen_mass
                mixed[word] = reserved_unseen_mass * (
                    (1.0 - weight) * base_component + weight * class_component
                )
        cache[cache_key] = mixed
        while len(cache) > CLASS_MIX_CACHE_MAX_ENTRIES:
            cache.popitem(last=False)
        return mixed

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

    def _class_distribution_for_context(self, context: tuple[str, ...]) -> dict[str, float]:
        """Smoothed distribution backed off through pooled class contexts."""
        if not context:
            counts = self.counts.get((), Counter())
            denom = sum(counts.values()) + self.alpha * len(self.vocab)
            return {word: (counts.get(word, 0) + self.alpha) / denom for word in self.vocab}
        base = self._class_distribution_for_context(context[1:])
        node = self.class_context_nodes.get(context)
        counts = (node.metadata.get("target_counts", {})
                  if node is not None else self.class_context_counts.get(context, Counter()))
        total = sum(counts.values())
        if not total:
            return base
        local_mass = total / (total + self.alpha)
        return {word: local_mass * counts.get(word, 0) / total + (1.0 - local_mass) * base[word]
                for word in self.vocab}

    def _class_probability_for_context(self, context: tuple[str, ...], target: str) -> float:
        """Return one pooled-class target probability without building a dense vocabulary map."""
        if not context:
            counts = self.counts.get((), Counter())
            denom = sum(counts.values()) + self.alpha * len(self.vocab)
            return (counts.get(target, 0) + self.alpha) / denom
        base = self._class_probability_for_context(context[1:], target)
        node = self.class_context_nodes.get(context)
        counts = (node.metadata.get("target_counts", {})
                  if node is not None else self.class_context_counts.get(context, Counter()))
        total = sum(counts.values())
        if not total:
            return base
        local_mass = total / (total + self.alpha)
        return (local_mass * counts.get(target, 0) / total
                + (1.0 - local_mass) * base)

    def _count_probability_for_context(self, context: tuple[str, ...], target: str) -> float:
        """Raw interpolated-count probability used by the exact flat fallback."""
        if not context:
            counts = self.counts.get((), Counter())
            denominator = sum(counts.values()) + self.alpha * len(self.vocab)
            return (counts.get(target, 0) + self.alpha) / denominator
        base = self._count_probability_for_context(context[1:], target)
        counts = self.counts.get(context, Counter())
        total = sum(counts.values())
        if not total:
            return base
        mass = total / (total + self.alpha)
        return mass * counts.get(target, 0) / total + (1.0 - mass) * base

    def evaluate_class_generalization(
        self, documents: Iterable[str], *, group_ids: Iterable[str] | None = None,
        bootstrap_samples: int = 1000,
    ) -> dict:
        """Compare class-on/off loss on test pairs unseen in HFN training.

        Both predictions use this checkpoint's identical word Observer state;
        only class mixing changes. Confidence intervals resample independent
        source groups, not tokens. A pair is unseen when its adjacent
        ``(previous_word, target_word)`` combination is absent from training.
        """
        documents = list(documents)
        if group_ids is None:
            groups = [str(index) for index in range(len(documents))]
        else:
            groups = list(group_ids)
            if len(groups) != len(documents):
                raise ValueError("group_ids must contain one identifier per document")
        if bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be nonnegative")
        trained_pairs = {
            (context[-1], target)
            for context, targets in self.raw_word_context_counts.items()
            if len(context) == 1 for target, count in targets.items() if count > 0
        }
        # group -> split -> [plain NLL, class NLL, events, distinct pairs]
        grouped: dict[str, dict[str, list]] = defaultdict(lambda: {
            "unseen": [0.0, 0.0, 0, set()],
            "seen": [0.0, 0.0, 0, set()],
        })
        totals = {"unseen": [0.0, 0.0, 0, set()], "seen": [0.0, 0.0, 0, set()]}
        for text, group in zip(documents, groups):
            for sentence in split_sentences(text):
                sequence = tokenize(sentence) + [BOUNDARY]
                for index, target in enumerate(sequence):
                    raw_context = tuple(sequence[max(0, index - self.max_order + 1):index])
                    if not raw_context:
                        continue
                    pair = (raw_context[-1], target)
                    split_name = "unseen" if pair not in trained_pairs else "seen"
                    context = tuple(self._map(word) for word in raw_context)
                    mapped_target = self._map(target)
                    plain_probability = self._weighted_probability(
                        context, mapped_target, apply_class_mix=False,
                        apply_character_mix=False,
                    )
                    class_probability = self._weighted_probability(
                        context, mapped_target, apply_character_mix=False,
                    )
                    plain_nll = -math.log(max(plain_probability, 1e-300))
                    class_nll = -math.log(max(class_probability, 1e-300))
                    for bucket in (grouped[group][split_name], totals[split_name]):
                        bucket[0] += plain_nll
                        bucket[1] += class_nll
                        bucket[2] += 1
                        bucket[3].add(pair)

        def summarize(split_name: str) -> dict:
            total_plain, total_class, count, pairs = totals[split_name]
            group_deltas = []
            for values in grouped.values():
                p_nll, c_nll, events, _pairs = values[split_name]
                if events:
                    group_deltas.append((c_nll - p_nll) / events)
            interval = None
            mean_delta = float(np.mean(group_deltas)) if group_deltas else None
            if group_deltas and bootstrap_samples:
                rng = np.random.default_rng(20261025)
                samples = [float(np.mean(rng.choice(group_deltas, size=len(group_deltas), replace=True)))
                           for _ in range(bootstrap_samples)]
                interval = [float(np.quantile(samples, 0.025)),
                            float(np.quantile(samples, 0.975))]
            return {
                "events": count,
                "distinct_adjacent_pairs": len(pairs),
                **({"distinct_unseen_adjacent_pairs": len(pairs)}
                   if split_name == "unseen" else {}),
                "plain_cross_entropy_nats": total_plain / count if count else None,
                "class_cross_entropy_nats": total_class / count if count else None,
                "paired_class_minus_plain_delta_nats_per_event": mean_delta,
                "source_group_bootstrap_95ci": interval,
                "source_groups": len(group_deltas),
            }

        return {
            "comparison": "same word Observer weights; class mixing is the only predictor difference",
            "pair_definition": "adjacent raw-token pair absent from training bigram counts",
            "unseen_pairs": summarize("unseen"),
            "seen_pairs": summarize("seen"),
        }

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

    def save(self, path: str | Path) -> None:
        payload = {"format": ("hpm-hierarchical-next-unit-v3" if self._raw_counts_complete
                              else "hpm-hierarchical-next-unit-v2"),
                   "max_order": self.max_order, "alpha": self.alpha,
                   "character_word_mix": self.character_word_mix,
                   "char_alpha": self.char_alpha, "min_word_count": self.min_word_count, "vocab": self.vocab,
                   "contexts": [{"context": list(k), "counts": dict(v)} for k, v in sorted(self.counts.items())],
                   "word_counts": dict(self.word_counts), "char_vocab": self.char_vocab,
                   "char_contexts": [{"context": list(k), "counts": dict(v)} for k, v in sorted(self.char_counts.items())],
                   "sentence_counts": dict(self.sentence_counts),
                   "observer_weights": {
                       "word": {str(order): self.observer.get_weight(node.id)
                                for order, node in self.word_level_nodes.items()} if self.observer else {},
                       "character": {str(order): self.observer.get_weight(node.id)
                                     for order, node in self.char_level_nodes.items()} if self.observer else {},
                       "classes": {class_id: self.observer.get_weight(class_id)
                                   for class_id in self.class_node_registry} if self.observer else {},
                   },
                   "training_manifest": self.training_manifest}
        if self.class_nodes_enabled:
            payload["class_nodes_enabled"] = True
            payload["class_tau"] = self.class_tau
            payload["class_lambda"] = self.class_lambda
            payload["class_lambdas"] = self.class_lambdas
            payload["class_min_observations"] = self.class_min_observations
            payload["class_profile_limit"] = self.class_profile_limit
            payload["class_follower_limit"] = self.class_follower_limit
            payload["class_labels"] = self.class_labels
            payload["class_profile_features"] = self.class_profile_features
            payload["class_profiles"] = {key: value.tolist() for key, value in self.class_profiles.items()}
            payload["class_profile_variances"] = {
                key: value.tolist() for key, value in self.class_profile_variances.items()
            }
            payload["class_selection"] = self.class_selection
        if self.l5_language_state is not None:
            payload["l5_language_state"] = self.l5_language_state
            payload["l5_calibration_document_hashes"] = sorted(self.l5_calibration_document_hashes)
        if self._raw_counts_complete:
            payload["raw_contexts"] = [
                {"context": list(k), "counts": dict(v)}
                for k, v in sorted(self.raw_word_context_counts.items())
            ]
        payload["training_document_hashes"] = sorted(self._training_document_hashes)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HierarchicalNextUnitModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        checkpoint_format = payload.get("format")
        if checkpoint_format not in {"hpm-hierarchical-next-unit-v1", "hpm-hierarchical-next-unit-v2",
                                     "hpm-hierarchical-next-unit-v3"}:
            raise ValueError("unsupported HPM language checkpoint format")
        if checkpoint_format == "hpm-hierarchical-next-unit-v3" and "raw_contexts" not in payload:
            raise ValueError("version-3 checkpoint is missing raw lexical context counts")
        model = cls(payload["max_order"], payload["alpha"], payload["min_word_count"],
                    class_nodes=bool(payload.get("class_nodes_enabled", False)),
                    class_tau=payload.get("class_tau", 0.55),
                    class_lambda=float(payload.get("class_lambda", 0.5)),
                    class_min_observations=int(payload.get("class_min_observations", 3)),
                    class_profile_limit=int(payload.get("class_profile_limit", 1500)),
                    class_follower_limit=int(payload.get("class_follower_limit", 2048)))
        model.class_labels = {str(key): str(value) for key, value in payload.get("class_labels", {}).items()}
        model.class_lambdas = {str(key): float(value)
                               for key, value in payload.get("class_lambdas", {}).items()}
        if not model.class_lambdas and model.class_lambda > 0.0:
            # Older opt-in checkpoints used one global mixture coefficient.
            model.class_lambdas = {
                class_id: model.class_lambda
                for class_id in set(model.class_labels.values())
            }
        model.class_profile_features = [str(value) for value in payload.get("class_profile_features", [])]
        model.class_profiles = {str(key): np.asarray(value, dtype=float)
                                for key, value in payload.get("class_profiles", {}).items()}
        model.class_profile_variances = {str(key): np.asarray(value, dtype=float)
                                         for key, value in payload.get("class_profile_variances", {}).items()}
        model.class_selection = payload.get("class_selection")
        model._class_membership_frozen = bool(model.class_labels) or model.class_nodes_enabled
        model.char_alpha = payload.get("char_alpha", model.alpha)
        model.character_word_mix = float(payload.get("character_word_mix", 0.0))
        if not 0.0 <= model.character_word_mix <= 1.0:
            raise ValueError("checkpoint character_word_mix must be between 0 and 1")
        model.vocab = payload["vocab"]
        model.word_to_index = {word: i for i, word in enumerate(model.vocab)}
        model.word_counts = Counter(payload["word_counts"])
        model.counts = defaultdict(Counter, {
            tuple(item["context"]): Counter(item["counts"]) for item in payload["contexts"]
        })
        if "raw_contexts" in payload:
            model.raw_word_context_counts = defaultdict(Counter, {
                tuple(item["context"]): Counter(item["counts"])
                for item in payload["raw_contexts"]
            })
            model._raw_counts_complete = True
        else:
            model.raw_word_context_counts = defaultdict(Counter, {
                context: Counter(counts) for context, counts in model.counts.items()
            })
            model._raw_counts_complete = model.min_word_count == 1
        model.char_vocab = payload["char_vocab"]
        model.char_counts = defaultdict(Counter, {
            tuple(item["context"]): Counter(item["counts"]) for item in payload["char_contexts"]
        })
        model.sentence_counts = Counter(payload.get("sentence_counts", {}))
        model.training_manifest = payload.get("training_manifest")
        if payload.get("l5_language_state") is not None:
            restored_l5 = PrequentialL5Mixer.from_state_dict(payload["l5_language_state"])
            expected_experts = ("observer", *(f"word_order_{order}" for order in range(model.max_order)))
            if restored_l5.experts != expected_experts:
                raise ValueError("checkpoint L5 language experts do not match max_order")
            order_alpha = float(payload["l5_language_state"].get("order_alpha", 0.1))
            if not math.isfinite(order_alpha) or order_alpha <= 0:
                raise ValueError("checkpoint L5 order_alpha must be finite and positive")
            model.l5_language_state = {**restored_l5.state_dict(), "order_alpha": order_alpha}
            hashes = payload.get("l5_calibration_document_hashes", [])
            if any(not isinstance(value, str) or len(value) != 64 for value in hashes):
                raise ValueError("checkpoint L5 calibration document hashes are invalid")
            model.l5_calibration_document_hashes = set(hashes)
        model._training_document_hashes = set(payload.get("training_document_hashes", []))
        model._rebuild_spelling_char_model()
        if model.class_nodes_enabled:
            if not model.class_profile_features and model.class_profiles:
                model._profiles_from_counts(model.raw_word_context_counts)
                if all(len(profile) == len(model._current_profile_features)
                       for profile in model.class_profiles.values()):
                    model.class_profile_features = list(model._current_profile_features)
            model._fit_class_nodes_from_counts(refit=False)
        model._build_hfn_nodes()
        model._initialize_pattern_observer()
        for layer, registry in (("word", model.word_level_nodes),
                                ("character", model.char_level_nodes)):
            for order, weight in payload.get("observer_weights", {}).get(layer, {}).items():
                node = registry.get(int(order))
                if node is not None:
                    model.observer.set_weight(node.id, weight)
        for class_id, weight in payload.get("observer_weights", {}).get("classes", {}).items():
            if class_id in model.class_node_registry:
                model.observer.set_weight(class_id, float(weight))
        return model

def run_pilot(corpus: str | Path | list[str | Path], output_dir: str | Path, seed: int = 17,
              max_order: int = 3, workers: int = 1, min_word_count: int = 1,
              *, class_nodes: bool = False) -> dict:
    """Train and evaluate the language pilot (implementation in language_pilot)."""
    from hpm_ai_v2.training.language_pilot import run_pilot as _run_pilot
    return _run_pilot(corpus, output_dir, seed, max_order, workers, min_word_count,
                      class_nodes=class_nodes)
