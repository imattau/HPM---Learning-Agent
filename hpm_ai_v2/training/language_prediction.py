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
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hpm_ai_v2.training.l5_language_meta import PrequentialL5Mixer

TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*|[^\w\s]", re.UNICODE)
BOUNDARY = "<eos>"
UNK = "<unk>"
SPELL_BOW = "<bow>"
SPELL_EOW = "</w>"
SPELL_UNK = "<unk-char>"
MAX_FIT_WORKERS = 4
CLASS_MAX_SEEN_REGRESSION_NATS = 0.05
CHARACTER_WORD_MIX_CANDIDATES = (0.0, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5)
CLASS_MIX_CACHE_MAX_ENTRIES = 32

_SWEEP_MODEL = None
_SWEEP_TEXTS = None
_EVAL_MODELS = None
_EVAL_DOCUMENTS = None


def _init_sweep_worker(model, texts):
    global _SWEEP_MODEL, _SWEEP_TEXTS
    _SWEEP_MODEL, _SWEEP_TEXTS = model, texts


def _evaluate_sweep_point(point):
    kind, alpha = point
    if kind == "characters":
        _SWEEP_MODEL.char_alpha = alpha
        return kind, alpha, _SWEEP_MODEL.evaluate_characters(_SWEEP_TEXTS)
    _SWEEP_MODEL.alpha = alpha
    metrics = (_SWEEP_MODEL.evaluate(_SWEEP_TEXTS, include_surface=False)
               if hasattr(_SWEEP_MODEL, "surface_word_probability")
               else _SWEEP_MODEL.evaluate(_SWEEP_TEXTS))
    return kind, alpha, metrics


def _init_evaluation_workers(models, documents):
    global _EVAL_MODELS, _EVAL_DOCUMENTS
    _EVAL_MODELS, _EVAL_DOCUMENTS = models, documents


def _evaluate_metric_job(job):
    name, model_key, metric_type, documents_key = job
    model = _EVAL_MODELS[model_key]
    documents = _EVAL_DOCUMENTS[documents_key]
    if metric_type == "characters":
        metrics = model.evaluate_characters(documents)
    else:
        metrics = model.evaluate(documents)
    return name, metrics


def _run_validation_sweeps(model, texts, points, workers, progress_callback=None):
    """Score independent smoothing values in bounded worker processes."""
    if workers <= 1 or len(points) < 2:
        results = []
        for kind, alpha in points:
            if kind == "characters":
                model.char_alpha = alpha
                metrics = model.evaluate_characters(texts)
            else:
                model.alpha = alpha
                metrics = (model.evaluate(texts, include_surface=False)
                           if hasattr(model, "surface_word_probability")
                           else model.evaluate(texts))
            result = (kind, alpha, metrics)
            results.append(result)
            if progress_callback:
                progress_callback(result, len(results), len(points))
        return results
    # Fork shares the fitted count tables copy-on-write and avoids serializing
    # the large HFN graph into every worker. Fall back to serial on platforms
    # without fork rather than silently changing the resource profile.
    if "fork" not in multiprocessing.get_all_start_methods():
        return _run_validation_sweeps(model, texts, points, 1)
    context = multiprocessing.get_context("fork")
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(points)), mp_context=context,
                             initializer=_init_sweep_worker, initargs=(model, texts)) as pool:
        futures = [pool.submit(_evaluate_sweep_point, point) for point in points]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if progress_callback:
                progress_callback(result, len(results), len(points))
    point_order = {point: index for index, point in enumerate(points)}
    results.sort(key=lambda result: point_order[(result[0], result[1])])
    return results


def _run_metric_jobs(models, documents, jobs, workers, progress_callback=None):
    """Evaluate independent held-out metrics in bounded forked workers."""
    if workers <= 1 or len(jobs) < 2 or "fork" not in multiprocessing.get_all_start_methods():
        results = []
        for job in jobs:
            name, model_key, metric_type, document_key = job
            model, texts = models[model_key], documents[document_key]
            metrics = model.evaluate_characters(texts) if metric_type == "characters" else model.evaluate(texts)
            result = (name, metrics)
            results.append(result)
            if progress_callback:
                progress_callback(result, len(results), len(jobs))
        return results
    context = multiprocessing.get_context("fork")
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context,
                             initializer=_init_evaluation_workers, initargs=(models, documents)) as pool:
        futures = [pool.submit(_evaluate_metric_job, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if progress_callback:
                progress_callback(result, len(results), len(jobs))
    return results


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def split_sentences(text: str) -> list[str]:
    """Split text into punctuation/newline-delimited units for local context."""
    return [part for part in re.split(r"(?<=[.!?])\s+|\n+", text) if tokenize(part)]


def _count_raw_text_batch(texts: list[str], max_order: int):
    """Count independent document-local lexical and character statistics."""
    tokenized = []
    word_counts = Counter()
    char_counts = defaultdict(Counter)
    for text in texts:
        sentence_tokens = [tokenize(sentence) for sentence in split_sentences(text)]
        tokenized.append(sentence_tokens)
        for tokens in sentence_tokens:
            word_counts.update(tokens)
        for sentence in split_sentences(text):
            chars = list(sentence.lower()) + [BOUNDARY]
            padded = [BOUNDARY] * (max_order - 1) + chars
            for i in range(max_order - 1, len(padded)):
                target = padded[i]
                for order in range(max_order):
                    char_counts[tuple(padded[i - order:i]) if order else ()][target] += 1
    return tokenized, word_counts, dict(char_counts)


def _batches(items: list[str], size: int = 16):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    source_label = Path(path).name
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                text = row.get("text")
                if isinstance(text, str) and text.strip():
                    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                    source_id = str(row.get("source_id", row.get("id", len(rows))))
                    group = metadata.get("group_id") or metadata.get("conversation_id") or metadata.get("message_tree_id")
                    if not group and all(metadata.get(key) for key in (
                        "huggingface_dataset_id", "huggingface_revision", "huggingface_split",
                        "huggingface_row_id",
                    )):
                        group = "|".join(str(metadata[key]) for key in (
                            "huggingface_dataset_id", "huggingface_revision", "huggingface_split",
                            "huggingface_row_id",
                        ))
                    if not group and metadata.get("source_path"):
                        group = f"{metadata['source_path']}:{metadata.get('document_index', 0)}"
                    group = group or metadata.get("parent_id") or source_id
                    source_path = str(metadata.get("source_path", ""))
                    source_type = metadata.get("source_type", "unknown")
                    if Path(source_path).suffix.lower() in {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".h", ".cpp", ".cs", ".rb", ".php", ".sh"}:
                        source_type = "code"
                    rows.append({"id": source_id, "group_id": f"{source_label}:{group}", "text": text,
                                 "source_file": source_label, "source_type": source_type,
                                 "source_path": source_path, "license": metadata.get("license", "unspecified"),
                                 "metadata": metadata})
    return rows


def split_documents(rows: list[dict], seed: int = 17) -> dict[str, list[dict]]:
    """Stable source-level split; small mixed-source corpora are stratified.

    With at most twelve independent sources, a plain 80/10/10 hash can leave
    a whole source type out of validation. When every source type has at least
    three groups, allocate each type approximately 60/20/20 by group so train,
    validation and test all cover the same source types. Larger corpora retain
    the usual 80/10/10 split. No rows from one source group are separated.
    """
    group_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        group_rows[str(row.get("group_id", row["id"]))].append(row)
    group_types = {
        group: {str(row.get("source_type", "unknown")) for row in group_members}
        for group, group_members in group_rows.items()
    }
    source_groups: dict[str, list[str]] = defaultdict(list)
    for group, types in group_types.items():
        if len(types) != 1:
            source_groups.clear()
            break
        source_groups[next(iter(types))].append(group)
    if (1 < len(source_groups) and len(group_rows) <= 12
            and all(len(groups) >= 3 for groups in source_groups.values())):
        buckets: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
        for source_type, groups in sorted(source_groups.items()):
            ordered = sorted(groups, key=lambda group: hashlib.sha256(
                f"{seed}:{source_type}:{group}".encode()).digest())
            train_count = max(1, int(0.6 * len(ordered)))
            validation_count = max(1, int(0.2 * len(ordered)))
            assignments = {
                "train": ordered[:train_count],
                "validation": ordered[train_count:train_count + validation_count],
                "test": ordered[train_count + validation_count:],
            }
            for name, assigned_groups in assignments.items():
                buckets[name].extend(row for group in assigned_groups for row in group_rows[group])
        return buckets

    buckets: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        digest = hashlib.sha256(f"{seed}:{row.get('group_id', row['id'])}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        name = "train" if value < .8 else "validation" if value < .9 else "test"
        buckets[name].append(row)
    # For tiny corpora, deterministic sorted fallback ensures all splits exist.
    if any(not buckets[name] for name in buckets):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[row.get("group_id", row["id"])].append(row)
        ordered_groups = sorted(groups.items(), key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).digest())
        n = len(ordered_groups)
        a, b = max(1, int(n * .8)), max(2, int(n * .9))
        buckets = {"train": [], "validation": [], "test": []}
        for name, group_slice in (("train", ordered_groups[:a]), ("validation", ordered_groups[a:b]), ("test", ordered_groups[b:])):
            buckets[name] = [row for _, group_rows in group_slice for row in group_rows]
    return buckets


class HierarchicalNextUnitModel:
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
        # Exact-text identity lets checkpoint continuation ignore epoch replay
        # instead of treating already-seen examples as independent evidence.
        self._training_document_hashes: set[str] = set()
        self._candidate_cache: dict[tuple[tuple[str, ...], int], frozenset[str]] = {}
        self.l5_language_state: dict | None = None
        self.l5_calibration_document_hashes: set[str] = set()

    def fit(
        self, documents: Iterable[str], workers: int = 1, *,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> "HierarchicalNextUnitModel":
        if workers < 1:
            raise ValueError("workers must be at least 1")
        def report(stage: str, **details) -> None:
            if progress_callback:
                progress_callback({"fit_stage": stage, **details})

        self._candidate_cache.clear()
        self._character_word_prior_cache = None
        texts = list(documents)
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
        if self.class_nodes_enabled:
            for node in self.word_level_nodes.values():
                self.observer.set_weight(node.id, 1.0)
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
        min_word_count: int | None = None,
        progress_callback: Callable[[dict], None] | None = None,
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
        replay_texts = [text for text in observer_replay_documents if text.strip()]
        if not supplied_texts:
            return self
        supplied_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest()
                           for text in supplied_texts]
        seen_hashes = set(self._training_document_hashes)
        texts = []
        new_hashes = []
        for text, digest in zip(supplied_texts, supplied_hashes):
            if digest not in seen_hashes:
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
        progress("rebuild_hfn_graph", raw_contexts=len(self.raw_word_context_counts),
                 mapped_contexts=len(self.counts))
        self._build_hfn_nodes()
        progress("recompute_observer_reliability", hfn_nodes=len(self.nodes))
        self._fit_pattern_observer()
        if self.class_nodes_enabled and self.class_lambda > 0.0:
            for node in self.word_level_nodes.values():
                self.observer.set_weight(node.id, 1.0)
        progress("train_observer_weights", objective="training_leave_one_event_out_next_unit_cross_entropy")
        word_training = self.train_observer_weights(
            [*texts, *replay_texts], character=False,
            progress_callback=lambda details: progress("train_observer_weights", **details),
        )
        character_training = self.train_observer_weights(
            [*texts, *replay_texts], character=True,
            progress_callback=lambda details: progress("train_observer_weights", **details),
        )
        if self.training_manifest is not None:
            batches_manifest = self.training_manifest.setdefault("incremental_training_batches", [])
            batches_manifest.append({
                "documents": len(texts),
                "observer_replay_documents": len(replay_texts),
                "min_word_count": self.min_word_count,
                "characters": sum(map(len, texts)),
                "text_sha256": hashlib.sha256("\n".join(sorted(texts)).encode()).hexdigest(),
                "observer_training": {"word": word_training, "character": character_training},
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
        """IDF-weighted positive PMI of each word's observed followers."""
        followers: dict[str, Counter] = defaultdict(Counter)
        for context, targets in raw_counts.items():
            if len(context) == 1 and context[0] not in (BOUNDARY, UNK):
                followers[context[0]].update(targets)
        eligible = {word: counts for word, counts in followers.items()
                    if sum(counts.values()) >= self.class_min_observations}
        if len(eligible) > self.class_profile_limit:
            eligible = dict(sorted(eligible.items(), key=lambda item: (-sum(item[1].values()), item[0]))[
                            :self.class_profile_limit])
        # Classes summarize predictable follower roles. A word with dozens of
        # unrelated followers is better retained as an exact context than
        # forced into a broad class (common articles are the typical case).
        eligible = {word: counts for word, counts in eligible.items()
                    if -sum((count / sum(counts.values())) *
                            math.log(count / sum(counts.values())) for count in counts.values())
                    <= math.log(5.0)}
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
        """Select tau/lambda from pair-held-out and ordinary held-out events."""
        contexts = sorted(context for context in self.raw_word_context_counts
                          if len(context) == max(0, self.max_order - 1))
        if not contexts:
            return None, 0.0
        unique_edges = sorted((context, target) for context in contexts
                              for target, count in self.raw_word_context_counts[context].items()
                              if count > 0)
        pairs = sorted({(context[-1], target) for context, target in unique_edges if context})
        if not pairs or len(unique_edges) < 4:
            return None, 0.0
        held_pairs = {pair for index, pair in enumerate(pairs)
                      if int.from_bytes(hashlib.sha256(f"pair:{pair}".encode()).digest()[:4], "big") % 10 == 0}
        if not held_pairs:
            held_pairs = {pairs[0]}
        ordinary = [edge for edge in unique_edges if edge[1] and
                    (not edge[0] or (edge[0][-1], edge[1]) not in held_pairs)
                    and self.raw_word_context_counts[edge[0]][edge[1]] >= 2]
        held_ordinary = {edge for edge in ordinary
                         if int.from_bytes(hashlib.sha256(f"ordinary:{edge}".encode()).digest()[:4], "big") % 7 == 0}
        ordinary_validation_counts = {
            edge: min(max(1, int(round(self.raw_word_context_counts[edge[0]][edge[1]] * 0.1))),
                      self.raw_word_context_counts[edge[0]][edge[1]] - 1)
            for edge in held_ordinary
        }
        validation = []
        for context, target in unique_edges:
            edge = (context, target)
            if context and (context[-1], target) in held_pairs:
                validation.append((context, target, self.raw_word_context_counts[context][target], 1.0))
            elif edge in ordinary_validation_counts:
                validation.append((context, target, ordinary_validation_counts[edge], 1.0))
        if not validation:
            return None, 0.0
        train_tables: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        for context, targets in self.raw_word_context_counts.items():
            for target, count in targets.items():
                edge = (context, target)
                held_count = count if (context and (context[-1], target) in held_pairs) else 0
                if len(context) == self.max_order - 1:
                    held_count = max(held_count, ordinary_validation_counts.get(edge, 0))
                if count > held_count:
                    train_tables[context][target] += count - held_count
        profiles = self._profiles_from_counts(train_tables)
        if not profiles:
            return None, 0.0
        candidates = [(None, 0.0)]
        # Favor coarse, reusable classes and keep lambda within the seen-pair
        # trust region observed in the frozen composition gates.
        candidates.extend((tau, lam) for tau in (0.7, 0.8, 0.9)
                          for lam in (0.025, 0.05, 0.1, 0.2, 0.35, 0.5))
        # Cluster membership depends on tau, not lambda. Reuse each clustering
        # across all six mixture strengths; otherwise the expensive pairwise
        # distance/linkage pass is needlessly repeated for every candidate.
        labels_by_tau: dict[float, dict[str, str]] = {}
        for tau in (0.7, 0.8, 0.9):
            labels: dict[str, str] = {}
            for index, members in enumerate(
                group for group in self._average_link_clusters(profiles, tau)
                if len(group) > 1
            ):
                label = f"validation_class_{index}"
                labels.update({word: label for word in members})
            labels_by_tau[tau] = labels
        best = (float("inf"), None, 0.0)
        no_class_score = None
        best_pair_improvements: dict[tuple[str, str], list[float]] = {}
        best_seen_delta = None
        scored_candidates = []
        candidate_diagnostics = []
        for tau, lam in candidates:
            labels = labels_by_tau.get(tau, {}) if lam > 0 else {}
            pooled: dict[tuple[str, ...], Counter] = defaultdict(Counter)
            for context, targets in train_tables.items():
                class_context = self._abstract_prediction_slot(context, labels)
                if context and class_context != context:
                    for target, count in targets.items():
                        pooled[class_context][self._map(target)] += count
            counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
            for context, targets in train_tables.items():
                mapped_context = tuple(self._map(word) for word in context)
                for target, count in targets.items():
                    counts[mapped_context][self._map(target)] += count

            def dist(context: tuple[str, ...], class_mode: bool = False) -> dict[str, float]:
                table = pooled if class_mode else counts
                if not context:
                    base_counts = counts.get((), Counter())
                    den = sum(base_counts.values()) + self.alpha * len(self.vocab)
                    return {word: (base_counts.get(word, 0) + self.alpha) / den for word in self.vocab}
                base = dist(context[1:], class_mode)
                local = table.get(context, Counter())
                total = sum(local.values())
                if not total:
                    return base
                mass = total / (total + self.alpha)
                return {word: mass * local.get(word, 0) / total + (1.0 - mass) * base[word]
                        for word in self.vocab}

            loss, observations = 0.0, 0
            seen_loss, seen_plain_loss, seen_observations = 0.0, 0.0, 0
            pair_improvements: dict[tuple[str, str], list[float]] = defaultdict(list)
            pair_losses: dict[tuple[str, str], list[float]] = defaultdict(list)
            for raw_context, target, frequency, validation_weight in validation:
                # Pair holdout is meaningful for class selection only where the
                # context word itself has a learned follower profile.
                if not raw_context or raw_context[-1] not in profiles:
                    continue
                follower_counts = train_tables.get((raw_context[-1],), Counter())
                follower_total = sum(follower_counts.values())
                if follower_total:
                    follower_probs = np.asarray(list(follower_counts.values()), dtype=float) / follower_total
                    entropy = -float(np.sum(follower_probs * np.log(follower_probs)))
                    # Broad function-word contexts are poor evidence for
                    # lexical compositional classes; keep ordinary validation
                    # from selecting a class only because it smooths articles.
                    if entropy > math.log(5.0):
                        continue
                is_seen_pair = bool(raw_context and (raw_context[-1], target) not in held_pairs)
                context = tuple(self._map(word) for word in raw_context)
                target = self._map(target)
                plain = dist(context)
                class_context = self._abstract_prediction_slot(raw_context, labels)
                if class_context != raw_context and class_context in pooled:
                    class_dist = dist(class_context, True)
                    # Validate the same direct expert blend used by inference.
                    probability = (1.0 - lam) * plain[target] + lam * class_dist[target]
                else:
                    probability = plain[target]
                loss -= validation_weight * frequency * math.log(max(probability, 1e-12))
                observations += validation_weight * frequency
                if is_seen_pair:
                    seen_loss -= validation_weight * frequency * math.log(max(probability, 1e-12))
                    seen_plain_loss -= validation_weight * frequency * math.log(max(plain[target], 1e-12))
                    seen_observations += validation_weight * frequency
                pair_improvements[(raw_context[-1], target)].append(
                    math.log(max(probability, 1e-12) / max(plain[target], 1e-12)))
                pair_losses[(raw_context[-1], target)].append(-math.log(max(probability, 1e-12)))
            score = loss / max(observations, 1)
            if tau is None:
                no_class_score = score
            seen_delta = ((seen_loss - seen_plain_loss) / seen_observations
                          if seen_observations else 0.0)
            if tau is not None and seen_delta > CLASS_MAX_SEEN_REGRESSION_NATS:
                continue
            edge_losses = np.asarray([float(np.mean(values)) for values in pair_losses.values()])
            candidate_se = (float(np.std(edge_losses, ddof=1) / math.sqrt(len(edge_losses)))
                            if len(edge_losses) > 1 else float("inf"))
            scored_candidates.append((score, candidate_se, tau, lam,
                                      pair_improvements, seen_delta,
                                      len({label for label in labels.values()})))
            candidate_diagnostics.append({"tau": tau, "lambda": lam,
                                          "validation_cross_entropy_nats": score,
                                          "edge_standard_error_nats": candidate_se,
                                          "seen_pair_loss_delta_nats": seen_delta,
                                          "class_count": len({label for label in labels.values()})})
            if score < best[0] - 1e-12:
                best = (score, tau, lam)
                best_pair_improvements = pair_improvements
                best_seen_delta = seen_delta
        # Within one standard error of the best validation risk, prefer the
        # weakest class mix to limit seen-context disruption and variance.
        if scored_candidates:
            min_score = min(row[0] for row in scored_candidates)
            best_row = min(scored_candidates, key=lambda row: row[0])
            threshold = min_score + best_row[1]
            selected_row = min((row for row in scored_candidates if row[0] <= threshold),
                               key=lambda row: (row[6], row[3], row[0]))
            best = (selected_row[0], selected_row[2], selected_row[3])
            best_pair_improvements = selected_row[4]
            best_seen_delta = selected_row[5]
        pair_deltas = np.asarray([float(np.mean(values)) for values in best_pair_improvements.values()])
        validation_se = (float(np.std(pair_deltas, ddof=1) / math.sqrt(len(pair_deltas)))
                         if len(pair_deltas) > 1 else float("inf"))
        self.class_selection = {"validation_cross_entropy_nats": best[0],
                                "best_candidate_tau": best[1],
                                "best_candidate_lambda": best[2],
                                "no_class_validation_cross_entropy_nats": no_class_score,
                                "validation_gain_nats": (no_class_score - best[0]
                                                          if no_class_score is not None else None),
                                "held_pairs": len(held_pairs),
                                "ordinary_held_events": len(held_ordinary),
                                "paired_edge_standard_error_nats": validation_se,
                                "selected_seen_pair_loss_delta_nats": best_seen_delta,
                                "candidate_diagnostics": candidate_diagnostics}
        # Use a one-standard-error rule over distinct held adjacent pairs to
        # avoid selecting classes on a small, noisy validation advantage.
        if best[1] is None:
            self.class_selection["selection_reason"] = "no-class fallback: simplest candidate within one standard error"
            self.class_selection["selected_validation_cross_entropy_nats"] = no_class_score
            return None, 0.0
        if no_class_score is not None and no_class_score - best[0] <= validation_se:
            self.class_selection["selection_reason"] = "no-class fallback: best gain within one paired-edge standard error"
            self.class_selection["selected_validation_cross_entropy_nats"] = no_class_score
            return None, 0.0
        self.class_selection["selection_reason"] = "best validation score"
        self.class_selection["selected_validation_cross_entropy_nats"] = best[0]
        return best[1], best[2]

    def _fit_class_nodes_from_counts(self, *, refit: bool) -> None:
        """Induce/freeze classes, refresh compact projections, and pool counts."""
        profiles = self._profiles_from_counts(self.raw_word_context_counts)
        if refit:
            self.class_tau, self.class_lambda = self._validation_class_choice()
            profiles = self._profiles_from_counts(self.raw_word_context_counts)
            clusters = self._average_link_clusters(profiles, self.class_tau)
            self.class_labels = {}
            for index, members in enumerate(group for group in clusters if len(group) > 1):
                class_id = f"language_class_{index:06d}"
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
        if not self.class_labels or self.class_lambda <= 0:
            return
        for raw_context, targets in self.raw_word_context_counts.items():
            if not raw_context:
                continue
            class_context = self._abstract_prediction_slot(raw_context, self.class_labels)
            if class_context == raw_context:
                continue
            for target, count in targets.items():
                self.class_context_counts[class_context][self._map(target)] += count

    def _build_hfn_nodes(
        self, *, progress_callback: Callable[[dict], None] | None = None,
    ) -> None:
        def report(stage: str, **details) -> None:
            if progress_callback:
                progress_callback({"fit_stage": "hfn_node_build", "node_stage": stage,
                                   **details})

        self.nodes.clear()
        self._class_mix_probability_cache = OrderedDict()
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
        for registry, counts_table in ((self.word_level_nodes, self.counts),
                                        (self.char_level_nodes, self.char_counts)):
            outcomes = [[0, 0] for _ in range(self.max_order)]
            for context, counts in counts_table.items():
                order = len(context)
                if order >= self.max_order:
                    continue
                for target, frequency in counts.items():
                    outcomes[order][1] += frequency
                    if self._leave_one_out_correct(counts, target):
                        outcomes[order][0] += frequency
            for order, (correct, total) in enumerate(outcomes):
                # Smoothed empirical reliability keeps unseen levels usable.
                reliability = (correct + 1.0) / (total + 2.0)
                self.observer.set_weight(registry[order].id, reliability)

    @staticmethod
    def _leave_one_out_correct(counts: Counter[str], target: str) -> bool:
        """Whether the modal prediction remains correct after removing this event."""
        if not counts:
            return False
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
        if not level_nodes or self.observer is None:
            if character:
                return self._char_distribution(context).get(target, 0.0)
            return self._distribution_for_context(context).get(target, 0.0)
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
            class_context = self._abstract_prediction_slot(context, self.class_labels)
            if class_context != context and class_context in self.class_context_nodes:
                weight_key = tuple(self.observer.get_weight(node.id)
                                   for _, node in sorted(level_nodes.items()))
                cache_key = (context, weight_key, self.class_lambda, self.alpha)
                cache = getattr(self, "_class_mix_probability_cache", None)
                if cache is None:
                    cache = self._class_mix_probability_cache = OrderedDict()
                mixed = cache.get(cache_key)
                if mixed is None:
                    plain = {word: self._weighted_probability(
                        context, word, apply_character_mix=False, apply_class_mix=False)
                             for word in self.vocab}
                    class_dist = self._class_distribution_for_context(class_context)
                    lam = self.class_lambda
                    mixed = {word: (1.0 - lam) * plain[word] + lam * class_dist[word]
                             for word in self.vocab}
                    cache[cache_key] = mixed
                    while len(cache) > CLASS_MIX_CACHE_MAX_ENTRIES:
                        cache.popitem(last=False)
                else:
                    cache.move_to_end(cache_key)
                probability = mixed.get(target, probability)
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

    def evaluate_characters(self, documents: Iterable[str]) -> dict:
        """Held-out next-character score from the same encoder-free learner."""
        vocab_size = len(self.char_vocab)
        nll = correct = count = 0
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
                    candidates = set()
                    probe = context
                    while True:
                        candidates.update(ch for ch, _ in self.char_counts.get(probe, Counter()).most_common(1))
                        if not probe:
                            break
                        probe = probe[1:]
                    ranked = sorted(candidates, key=lambda ch: self._char_probability(context, ch, vocab_size), reverse=True)
                    correct += int(bool(ranked) and ranked[0] == target)
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

    def _l5_order_expert_probabilities(
        self, context: tuple[str, ...], target: str, *, order_alpha: float = 0.1,
    ) -> dict[str, float]:
        """Score target under the fixed Observer and raw order experts."""
        vocabulary_size = max(1, len(self.vocab))
        result = {"observer": self._probability(context, target)}
        for order in range(self.max_order):
            suffix = context[-order:] if order else ()
            counts = self.counts.get(suffix, Counter())
            total = sum(counts.values())
            result[f"word_order_{order}"] = (
                counts.get(target, 0) + order_alpha
            ) / (total + order_alpha * vocabulary_size)
        return result

    def _new_l5_mixer(self, eta: float, fixed_share: float = 0.0) -> PrequentialL5Mixer:
        names = ("observer", *(f"word_order_{order}" for order in range(self.max_order)))
        mixer = PrequentialL5Mixer(names, eta=eta, fixed_share=fixed_share)
        # Keep most initial mass on the already calibrated predictor; L5 must
        # earn a shift toward raw order experts from prequential evidence.
        mixer.log_weights["observer"] = math.log(0.8)
        raw_prior = 0.2 / self.max_order
        for name in names[1:]:
            mixer.log_weights[name] = math.log(raw_prior)
        return mixer

    def fit_l5_meta(
        self, validation_documents: Iterable[str], *, eta: float = 0.1,
        order_alpha: float = 0.1, fixed_share: float = 0.0,
    ) -> dict:
        """Calibrate L5 over word-order experts on validation, prequentially.

        Each event is scored before its target updates the meta-weights. The
        base HFN count tables remain frozen; this only calibrates which context
        order is reliable for later predictions.
        """
        if not math.isfinite(order_alpha) or order_alpha <= 0:
            raise ValueError("order_alpha must be finite and positive")
        mixer = self._new_l5_mixer(eta, fixed_share=fixed_share)
        nll = count = 0
        documents = list(validation_documents)
        if not any(tokenize(sentence) for text in documents for sentence in split_sentences(text)):
            raise ValueError("L5 validation calibration requires at least one lexical event")
        for text in documents:
            for sentence in split_sentences(text):
                sequence = [self._map(word) for word in tokenize(sentence)] + [BOUNDARY]
                for index, target in enumerate(sequence):
                    context = tuple(sequence[max(0, index - self.max_order + 1):index])
                    probabilities = self._l5_order_expert_probabilities(
                        context, target, order_alpha=order_alpha,
                    )
                    probability = mixer.probability(probabilities)
                    nll -= math.log(max(probability, 1e-300))
                    mixer.update_after_score(probabilities)
                    count += 1
        self.l5_language_state = {**mixer.state_dict(), "order_alpha": float(order_alpha)}
        self.l5_calibration_document_hashes = {
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in documents
        }
        mean_loss = nll / max(count, 1)
        return {"events": count, "prequential_cross_entropy_nats": mean_loss,
                "prequential_perplexity": math.exp(min(mean_loss, 700)),
                "weights": mixer.weights(), "state": self.l5_language_state,
                "calibration_split": "validation"}

    def continue_l5_meta(self, new_validation_documents: Iterable[str]) -> dict:
        """Continue L5 reliability learning from exact-text-new validation data.

        This carries forward the selected L5 weights and learning rate. It is
        intended for a later data round after HFN training has incorporated the
        new training split; validation targets update L5 only after scoring.
        """
        if self.l5_language_state is None:
            raise RuntimeError("fit or tune L5 before continuing its calibration")
        documents = list(new_validation_documents)
        hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in documents]
        if not documents or len(set(hashes)) != len(hashes):
            raise ValueError("new L5 calibration documents must be nonempty and unique")
        if self.l5_calibration_document_hashes.intersection(hashes):
            raise ValueError("L5 calibration rejects exact-text replay from prior rounds")
        mixer = PrequentialL5Mixer.from_state_dict(self.l5_language_state)
        order_alpha = float(self.l5_language_state.get("order_alpha", 0.1))
        nll = count = 0
        for text in documents:
            for sentence in split_sentences(text):
                sequence = [self._map(word) for word in tokenize(sentence)] + [BOUNDARY]
                for index, target in enumerate(sequence):
                    context = tuple(sequence[max(0, index - self.max_order + 1):index])
                    probabilities = self._l5_order_expert_probabilities(
                        context, target, order_alpha=order_alpha,
                    )
                    probability = mixer.probability(probabilities)
                    nll -= math.log(max(probability, 1e-300))
                    mixer.update_after_score(probabilities)
                    count += 1
        if count == 0:
            raise ValueError("new L5 calibration data must contain a lexical event")
        previous_updates = int(self.l5_language_state["updates"])
        self.l5_language_state = {**mixer.state_dict(), "order_alpha": order_alpha}
        self.l5_calibration_document_hashes.update(hashes)
        mean_loss = nll / count
        return {"events": count, "previous_updates": previous_updates,
                "cumulative_updates": mixer.updates,
                "prequential_cross_entropy_nats": mean_loss,
                "weights": mixer.weights(), "state": self.l5_language_state,
                "new_validation_documents": len(documents)}

    def tune_l5_meta(
        self, validation_documents: Iterable[str], *,
        eta_candidates: Iterable[float] = (0.0001, 0.0003, 0.001, 0.003, 0.01),
        order_alpha_candidates: Iterable[float] = (0.1, 1.0, 10.0),
        fixed_share_candidates: Iterable[float] = (0.0, 0.02),
    ) -> dict:
        """Select L5 update settings by validation prequential log loss only.

        The fixed-share candidate set includes the legacy no-share behavior and
        a small share rate. Direct mixer construction and ``fit_l5_meta`` keep
        fixed share disabled unless it is explicitly requested.
        """
        documents = list(validation_documents)
        candidates = tuple(sorted(set(float(value) for value in eta_candidates)))
        alpha_candidates = tuple(sorted(set(float(value) for value in order_alpha_candidates)))
        share_candidates = tuple(sorted(set(float(value) for value in fixed_share_candidates)))
        if not candidates or any(not math.isfinite(value) or value <= 0 for value in candidates):
            raise ValueError("eta_candidates must contain finite positive values")
        if not alpha_candidates or any(not math.isfinite(value) or value <= 0 for value in alpha_candidates):
            raise ValueError("order_alpha_candidates must contain finite positive values")
        if not share_candidates or any(not math.isfinite(value) or not 0.0 <= value < 1.0
                                       for value in share_candidates):
            raise ValueError("fixed_share_candidates must contain finite values in [0, 1)")
        fixed_loss = self.score_cross_entropy(documents)
        sweep = []
        best = None
        for eta in candidates:
            for order_alpha in alpha_candidates:
                for fixed_share in share_candidates:
                    result = self.fit_l5_meta(documents, eta=eta, order_alpha=order_alpha,
                                              fixed_share=fixed_share)
                    item = {"eta": eta, "order_alpha": order_alpha,
                        "fixed_share": fixed_share,
                        "prequential_cross_entropy_nats": result["prequential_cross_entropy_nats"],
                        "weights": result["weights"]}
                    sweep.append(item)
                    if best is None or item["prequential_cross_entropy_nats"] < best["prequential_cross_entropy_nats"]:
                        best = item
                        selected_state = result["state"]
        self.l5_language_state = selected_state
        return {"fixed_observer_validation_cross_entropy_nats": fixed_loss,
                "selected_eta": best["eta"],
                "selected_order_alpha": best["order_alpha"],
                "selected_fixed_share": best["fixed_share"],
                "selected_prequential_cross_entropy_nats": best["prequential_cross_entropy_nats"],
                "sweep": sweep, "selected_state": selected_state,
                "selection_split": "validation"}

    def evaluate_l5_prequential(
        self, documents: Iterable[str], *, state: dict | None = None,
    ) -> dict:
        """Score a stream with L5 weights, updating only after each prediction.

        The supplied or checkpoint state is copied into a local mixer. Returned
        state can be carried to a later stream, while evaluation never mutates
        the fitted checkpoint implicitly.
        """
        if self.l5_language_state is None and state is None:
            raise RuntimeError("fit L5 on validation data or provide a saved L5 state first")
        mixer = PrequentialL5Mixer.from_state_dict(
            state if state is not None else self.l5_language_state
        )
        order_alpha = float((state if state is not None else self.l5_language_state).get("order_alpha", 0.1))
        nll = count = 0
        for text in documents:
            for sentence in split_sentences(text):
                sequence = [self._map(word) for word in tokenize(sentence)] + [BOUNDARY]
                for index, target in enumerate(sequence):
                    context = tuple(sequence[max(0, index - self.max_order + 1):index])
                    probabilities = self._l5_order_expert_probabilities(
                        context, target, order_alpha=order_alpha,
                    )
                    probability = mixer.probability(probabilities)
                    nll -= math.log(max(probability, 1e-300))
                    mixer.update_after_score(probabilities)
                    count += 1
        mean_loss = nll / max(count, 1)
        return {"events": count, "prequential_cross_entropy_nats": mean_loss,
                "prequential_perplexity": math.exp(min(mean_loss, 700)),
                "weights": mixer.weights(), "state": mixer.state_dict(),
                "starting_updates": int((state if state is not None else self.l5_language_state)["updates"]),
                "ending_updates": mixer.updates}

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
        class_context = self._abstract_prediction_slot(context, self.class_labels)
        if (class_context == context or self.class_lambda <= 0
                or class_context not in self.class_context_nodes):
            return plain
        class_dist = self._class_distribution_for_context(class_context)
        lam = self.class_lambda
        return {word: (1.0 - lam) * plain[word] + lam * class_dist[word]
                for word in self.vocab}

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
                   },
                   "training_manifest": self.training_manifest}
        if self.class_nodes_enabled:
            payload["class_nodes_enabled"] = True
            payload["class_tau"] = self.class_tau
            payload["class_lambda"] = self.class_lambda
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
        return model


def run_pilot(corpus: str | Path | list[str | Path], output_dir: str | Path, seed: int = 17,
              max_order: int = 3, workers: int = 1, min_word_count: int = 1,
              *, class_nodes: bool = False) -> dict:
    """Train and evaluate against n-gram baselines, optionally enabling HFN classes.

    The held-out HPM report always includes both vocabulary-mapped and exact
    surface-word cross-entropy. Class-node runs retain the same evaluation and
    additionally record the training-only class selection decision.
    """
    from hpm_ai_v2.training.ngram_baseline import NGramBaseline
    if workers < 1 or min_word_count < 1:
        raise ValueError("workers and min_word_count must be at least 1")
    corpus_paths = [corpus] if isinstance(corpus, (str, Path)) else list(corpus)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def status(stage: str, state: str, **details) -> None:
        payload = {"stage": stage, "state": state, "elapsed_seconds": round(time.monotonic() - started, 3), **details}
        temporary = out / "run_status.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(out / "run_status.json")

    status("read_corpora", "running", corpora=[str(path) for path in corpus_paths])
    rows = [row for path in corpus_paths for row in read_jsonl(path)]
    seen_texts: set[str] = set()
    unique_rows = []
    for row in rows:
        digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if digest not in seen_texts:
            seen_texts.add(digest)
            unique_rows.append(row)
    rows = unique_rows
    split = split_documents(rows, seed)
    status("read_corpora", "complete", source_rows=len(rows),
           split_documents={name: len(items) for name, items in split.items()})
    status("fit", "running", source_rows=len(rows), split_documents={name: len(items) for name, items in split.items()})
    def fit_progress(details: dict) -> None:
        fit_state = details.pop("state", "running")
        status("fit", fit_state, **details)

    model = HierarchicalNextUnitModel(max_order=max_order, min_word_count=min_word_count,
                                      class_nodes=class_nodes).fit(
        (row["text"] for row in split["train"]), workers=workers,
        progress_callback=fit_progress,
    )
    # Class-enabled fitting temporarily uses flat word weights while class
    # selection is unresolved. If validation rejects every class, restore the
    # ordinary HFN Observer initialization before the shared training pass.
    if class_nodes and model.class_lambda <= 0.0:
        model._fit_pattern_observer()
    training_groups = sorted({row.get("group_id", row["id"]) for row in split["train"]})
    model.training_manifest = {
        "corpora": [str(path) for path in corpus_paths], "seed": seed,
        "split_documents": {name: len(items) for name, items in split.items()},
        "train_group_count": len(training_groups),
        "train_group_ids_sha256": hashlib.sha256("\n".join(training_groups).encode()).hexdigest(),
        "train_source_types": dict(Counter(row["source_type"] for row in split["train"])),
        "train_licenses": dict(Counter(row["license"] for row in split["train"])),
        "objective": "encoder-free character and word next-unit prediction; normalized character spelling likelihood calibrated on validation and composed into exact next-word probabilities; HFN Observer weights fitted on training-only leave-one-event-out loss",
        "tokenizer": "unicode-regex-word-punctuation-lowercase-v1",
        "min_word_count": min_word_count,
    }
    if class_nodes:
        model.training_manifest["class_nodes_enabled"] = True
    train_texts = [row["text"] for row in split["train"]]
    status("observer_training", "running", training_documents=len(train_texts), workers=1)
    word_observer_training = model.train_observer_weights(
        train_texts,
        progress_callback=lambda details: status(
            "observer_training", "running", training_documents=len(train_texts), **details),
    )
    character_observer_training = model.train_observer_weights(
        train_texts, character=True,
        progress_callback=lambda details: status(
            "observer_training", "running", training_documents=len(train_texts), **details),
    )
    model.training_manifest["observer_training"] = {
        "word": word_observer_training,
        "character": character_observer_training,
    }
    model.save(out / "hpm_language_checkpoint.json")
    status("fit", "complete", source_rows=len(rows), split_documents={name: len(items) for name, items in split.items()},
           nodes=len(model.nodes), checkpoint="hpm_language_checkpoint.json")
    validation_texts = [row["text"] for row in split["validation"]]
    word_alphas = (0.1, 1, 10, 30, 100, 200, 300)
    char_alphas = (0.01, 0.1, 1, 10, 30, 100)
    status("observer_validation_calibration", "running", workers=workers)
    hpm_tuning = {}
    selected_word_alpha = model.alpha
    for completed, alpha in enumerate(word_alphas, 1):
        model.alpha = alpha
        hpm_tuning[str(alpha)] = model.score_cross_entropy(validation_texts)
        status("word_alpha_calibration", "running", workers=workers,
               completed_points=completed, total_points=len(word_alphas), last_alpha=alpha,
               last_validation_cross_entropy=hpm_tuning[str(alpha)])
    selected_word_alpha = float(min(hpm_tuning, key=lambda value: hpm_tuning[value]))
    model.alpha = selected_word_alpha
    hpm_validation = model.evaluate(validation_texts)
    char_tuning = {}
    selected_char_alpha = model.char_alpha
    for completed, alpha in enumerate(char_alphas, 1):
        model.char_alpha = alpha
        char_tuning[str(alpha)] = model.score_cross_entropy(validation_texts, character=True)
        status("character_alpha_calibration", "running", workers=workers,
               completed_points=completed, total_points=len(char_alphas), last_alpha=alpha,
               last_validation_cross_entropy=char_tuning[str(alpha)])
    selected_char_alpha = float(min(char_tuning, key=lambda value: char_tuning[value]))
    model.char_alpha = selected_char_alpha
    status("character_to_word_composition_calibration", "running", workers=workers)
    character_word_tuning = model.tune_character_word_mix(
        validation_texts,
        progress_callback=lambda details: status(
            "character_to_word_composition_calibration", "running", workers=workers, **details),
    )
    hpm_validation = model.evaluate(validation_texts)
    character_validation = model.evaluate_characters(validation_texts)
    model.save(out / "hpm_language_checkpoint.json")
    status("baseline_validation_and_heldout", "running", selected_hpm_alpha=selected_word_alpha,
           selected_character_alpha=selected_char_alpha,
           character_word_mix=character_word_tuning["selected_mix"])
    result = {"corpora": [str(path) for path in corpus_paths], "seed": seed, "workers": workers,
              "split_documents": {k: len(v) for k, v in split.items()},
              "source_rows_after_exact_deduplication": len(rows),
              "min_word_count": min_word_count,
              "source_rows_by_type": dict(Counter(row["source_type"] for row in rows)),
              "source_rows_by_license": dict(Counter(row["license"] for row in rows)),
              "selected_hpm_alpha": selected_word_alpha, "validation_hpm_alpha_cross_entropy": hpm_tuning,
              "selected_character_alpha": selected_char_alpha, "validation_character_alpha_cross_entropy": char_tuning,
              "character_to_word_composition": character_word_tuning,
              "observer_training": {"word": word_observer_training,
                                    "character": character_observer_training},
              "observer_validation_tuning": {
                  "weight_fit_partition": "training only",
                  "validation_used_for_weight_fit": False,
              },
              "nodes": {"characters": len(model.char_nodes), "words": len(model.word_nodes),
                        "character_contexts": len(model.char_context_nodes),
                        "phrase_contexts": len(model.phrase_nodes),
                        "sentence_patterns": len(model.sentence_nodes)},
              "observer_weights": {
                  "word_context_orders": {str(order): model.observer.get_weight(node.id)
                                          for order, node in model.word_level_nodes.items()},
                  "character_context_orders": {str(order): model.observer.get_weight(node.id)
                                               for order, node in model.char_level_nodes.items()},
                  "observer_level_nodes": len(model.word_level_nodes) + len(model.char_level_nodes),
                  "registered_hfn_nodes": len(model.pattern_forest) if model.pattern_forest else 0,
                  "weighted_pattern_links": sum(len(node.children()) for node in
                                                 [*model.word_level_nodes.values(),
                                                  *model.char_level_nodes.values()]),
              },
              "hpm": {"validation": hpm_validation},
              "character_model": {"validation": character_validation},
              "baselines": {}}
    if class_nodes:
        result["class_node_training"] = {
            "enabled": True,
            "selected_tau": model.class_tau,
            "selected_lambda": model.class_lambda,
            "class_count": len(model.class_node_registry),
            "member_word_count": len(model.class_labels),
            "selection": model.class_selection,
            "test_metrics_use": "hpm.test includes vocabulary_mapped and exact_surface_word_and_eos target spaces",
        }
    baseline_models = {}
    for order in range(1, max_order + 1):
        baseline = NGramBaseline(order=order, min_word_count=min_word_count).fit(
            row["text"] for row in split["train"]
        )
        baseline_alphas = (0.01, 0.1, 1, 10, 100)
        status("baseline_validation_sweep", "running", baseline_order=order, workers=workers)
        tuning = {}
        for completed, alpha in enumerate(baseline_alphas, 1):
            baseline.alpha = alpha
            tuning[str(alpha)] = baseline.score_cross_entropy(validation_texts)
            status(f"{order}_gram_validation_sweep", "running", workers=workers,
                   completed_points=completed, total_points=len(baseline_alphas),
                   last_alpha=alpha, last_validation_cross_entropy=tuning[str(alpha)])
        baseline.alpha = float(min(tuning, key=tuning.get))
        validation_metrics = baseline.evaluate(validation_texts)
        result["baselines"][f"{order}-gram"] = {"selected_alpha": baseline.alpha, "validation_alpha_cross_entropy": tuning,
                                                 "validation": validation_metrics}
        baseline_models[f"{order}-gram"] = baseline
    interpolated = NGramBaseline(order=max_order, smoothing="interpolated",
                                 min_word_count=min_word_count).fit(
        row["text"] for row in split["train"]
    )
    interpolated_alphas = (0.1, 1, 10, 30, 100, 200, 300)
    status("interpolated_ngram_validation_sweep", "running", workers=workers)
    interpolated_tuning = {}
    for completed, alpha in enumerate(interpolated_alphas, 1):
        interpolated.alpha = alpha
        interpolated_tuning[str(alpha)] = interpolated.score_cross_entropy(validation_texts)
        status("interpolated_ngram_validation_sweep", "running", workers=workers,
               completed_points=completed, total_points=len(interpolated_alphas),
               last_alpha=alpha, last_validation_cross_entropy=interpolated_tuning[str(alpha)])
    interpolated.alpha = float(min(interpolated_tuning, key=interpolated_tuning.get))
    interpolated_validation = interpolated.evaluate(validation_texts)
    result["interpolated_ngram_control"] = {
        "order": max_order, "selected_alpha": interpolated.alpha,
        "validation_alpha_cross_entropy": interpolated_tuning,
        "validation": interpolated_validation,
    }
    model.save(out / "hpm_language_checkpoint.json")
    restored = HierarchicalNextUnitModel.load(out / "hpm_language_checkpoint.json")
    test_rows = split["test"]
    documents = {"test": [row["text"] for row in test_rows]}
    models = {"hpm": model, "restored": restored, "characters": model, "interpolated": interpolated,
              **baseline_models}
    jobs = [
        ("hpm_test", "hpm", "words", "test"),
        ("character_test", "characters", "characters", "test"),
        ("restored_test", "restored", "words", "test"),
        ("restored_character_test", "restored", "characters", "test"),
        ("interpolated_test", "interpolated", "words", "test"),
    ]
    for order in range(1, max_order + 1):
        jobs.append((f"{order}_gram_test", f"{order}-gram", "words", "test"))
    source_types = sorted({row["source_type"] for row in test_rows})
    for source_type in source_types:
        documents[f"test_{source_type}"] = [row["text"] for row in test_rows if row["source_type"] == source_type]
        jobs.append((f"hpm_test_{source_type}", "hpm", "words", f"test_{source_type}"))
    status("heldout_evaluation", "running", workers=workers, metrics=len(jobs))

    def metric_progress(result, completed: int, total: int) -> None:
        name, metrics = result
        status("heldout_evaluation", "running", workers=workers, completed_metrics=completed,
               total_metrics=total, last_metric=name, last_tokens=metrics.get("tokens", metrics.get("characters")))

    metric_results = dict(_run_metric_jobs(models, documents, jobs, workers, metric_progress))
    result["hpm"]["test"] = metric_results["hpm_test"]
    result["character_model"]["test"] = metric_results["character_test"]
    result["interpolated_ngram_control"]["test"] = metric_results["interpolated_test"]
    hpm_test = result["hpm"]["test"]
    control_test = result["interpolated_ngram_control"]["test"]
    result["comparison_to_interpolated_ngram"] = {
        "cross_entropy_delta_nats_hpm_minus_control": (
            hpm_test["cross_entropy_nats"] - control_test["cross_entropy_nats"]
        ),
        "perplexity_ratio_hpm_over_control": hpm_test["perplexity"] / control_test["perplexity"],
        "top1_accuracy_delta_hpm_minus_control": hpm_test["top1_accuracy"] - control_test["top1_accuracy"],
        "top5_accuracy_delta_hpm_minus_control": hpm_test["top5_accuracy"] - control_test["top5_accuracy"],
        "same_selected_alpha": selected_word_alpha == interpolated.alpha,
        "predictively_equivalent_within_1e-12": all(
            abs(hpm_test[key] - control_test[key]) <= 1e-12
            for key in ("cross_entropy_nats", "perplexity", "top1_accuracy", "top5_accuracy")
        ),
        "interpretation": (
            "HFN Observer utility scales each order's evidence-supported backoff mass; unit weights recover the "
            "interpolated count control. The held-out deltas measure whether supervised Observer calibration "
            "improves on that control."
        ),
    }
    result["hpm"]["test_by_source_type"] = {
        source_type: metric_results[f"hpm_test_{source_type}"] for source_type in source_types
    }
    for order in range(1, max_order + 1):
        result["baselines"][f"{order}-gram"]["test"] = metric_results[f"{order}_gram_test"]
    result["checkpoint_roundtrip"] = {
        "word_test_metrics_match": result["hpm"]["test"] == metric_results["restored_test"],
        "character_test_metrics_match": result["character_model"]["test"] == metric_results["restored_character_test"],
        "restored_hfn_nodes": len(restored.nodes),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    status("complete", "complete", elapsed_seconds=round(time.monotonic() - started, 3), metrics="metrics.json")
    return result
