"""Training and held-out evaluation orchestration for the language pilot."""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from hpm_ai_v2.training.language_prediction import (
    HierarchicalNextUnitModel, _run_metric_jobs, _run_validation_sweeps,
    read_jsonl, split_documents,
)

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
    # Small class-enabled runs may not have enough distinct documents for an
    # internal calibration split; keep the legacy training path for those.
    if (class_nodes and model.class_lambda <= 0.0
            and model.internal_observer_calibration is None):
        model._set_pattern_observer_reliability()
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
    if model.internal_observer_calibration is not None:
        model.training_manifest["internal_observer_calibration"] = \
            model.internal_observer_calibration
    train_texts = [row["text"] for row in split["train"]]
    status("observer_training", "running", training_documents=len(train_texts), workers=1)
    if model.internal_observer_calibration is not None:
        word_observer_training = {
            "unit": "word",
            "objective": "heldout_document_validation_next_unit_cross_entropy",
            **model.internal_observer_calibration,
        }
    else:
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
    if class_nodes:
        test_texts = [row["text"] for row in test_rows]
        test_groups = [row.get("group_id", row["id"]) for row in test_rows]
        result["class_node_training"]["source_grouped_test_pair_generalization"] = \
            model.evaluate_class_generalization(test_texts, group_ids=test_groups)
        result["class_node_training"]["checkpoint_pair_generalization_matches"] = (
            result["class_node_training"]["source_grouped_test_pair_generalization"]
            == restored.evaluate_class_generalization(test_texts, group_ids=test_groups)
        )
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
