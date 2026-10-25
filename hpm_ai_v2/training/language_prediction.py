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
from hpm_ai_v2.training.language_checkpoint import LanguageCheckpointMixin
from hpm_ai_v2.training.language_evaluation import LanguageEvaluationMixin
from hpm_ai_v2.training.language_class_nodes import (
    CLASS_MAX_SEEN_REGRESSION_NATS, CLASS_RESERVED_UNSEEN_MASS,
    CLASS_MAX_MEMBERS, CLASS_MIN_COHESION, CLASS_MIN_VALIDATION_PAIRS,
    CLASS_LAMBDA_GRID, CLASS_MIX_CACHE_MAX_ENTRIES,
    CLASS_MIX_MASS_CACHE_MAX_ENTRIES, LanguageClassMixin,
)

SPELL_BOW = "<bow>"
SPELL_EOW = "</w>"
SPELL_UNK = "<unk-char>"
MAX_FIT_WORKERS = 4
CHARACTER_WORD_MIX_CANDIDATES = (0.0, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5)

from hpm_ai_v2.training.language_constants import BOUNDARY, TOKEN_RE, UNK
from hpm_ai_v2.training.language_corpus import (
    _batches, _count_raw_text_batch, read_jsonl, split_documents, split_sentences, tokenize,
)
from hpm_ai_v2.training.language_l5_mixin import LanguageL5Mixin as _LanguageL5Mixin
from hpm_ai_v2.training.language_validation_workers import (
    _run_metric_jobs, _run_validation_sweeps,
)

class HierarchicalNextUnitModel(LanguageCheckpointMixin, LanguageEvaluationMixin, LanguageClassMixin, _LanguageL5Mixin):
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
















def run_pilot(corpus: str | Path | list[str | Path], output_dir: str | Path, seed: int = 17,
              max_order: int = 3, workers: int = 1, min_word_count: int = 1,
              *, class_nodes: bool = False) -> dict:
    """Train and evaluate the language pilot (implementation in language_pilot)."""
    from hpm_ai_v2.training.language_pilot import run_pilot as _run_pilot
    return _run_pilot(corpus, output_dir, seed, max_order, workers, min_word_count,
                      class_nodes=class_nodes)
