"""Evaluate HFN class nodes on source-grouped real-text unseen word pairs.

Class geometry is selected on half of the source-grouped validation split.
Word-order Observer tuning and per-class reinforcement use a disjoint, bounded
set of complete validation groups. Test groups are scored once after all
selection and calibration. The predictor comparison changes only class mixing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from collections import defaultdict
from pathlib import Path

# The evaluator builds and scores one model in one process. Keep BLAS from
# silently starting a second pool of workers on large HFN graphs.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np

from hpm_ai_v2.training.language_prediction import (
    HierarchicalNextUnitModel,
    read_jsonl,
    split_documents,
    split_sentences,
    tokenize,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS = {
    "oasst1_2500": ROOT / "data/pretraining/oasst1_train_en_2500.jsonl",
    "wikitext2_1500": ROOT / "data/pretraining/wikitext2_raw_train_sample_1500.jsonl",
}


def _group_key(row: dict) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("group_id") or metadata.get("conversation_id")
               or metadata.get("message_tree_id") or row.get("group_id") or row.get("id"))


def _tokens(rows: list[dict]) -> int:
    return sum(len(tokenize(sentence)) + 1 for row in rows
               for sentence in split_sentences(row["text"]))


def _class_quality(model: HierarchicalNextUnitModel) -> dict:
    members: dict[str, list[str]] = defaultdict(list)
    for word, class_id in model.class_labels.items():
        members[class_id].append(word)
    profiles = model._profiles_from_counts(model.raw_word_context_counts)
    classes = []
    for class_id, words in members.items():
        vectors = np.asarray([
            model._project_follower_profile(
                profiles.get(word, np.zeros(len(model.class_profile_features))),
                model.class_profile_features,
            ) for word in words
        ])
        normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        centroid = normalized.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        classes.append({
            "class_id": class_id,
            "size": len(words),
            "observer_weight": model.class_lambdas.get(class_id, 0.0),
            "mean_cosine_to_centroid": float(np.mean(normalized @ centroid)),
            "examples": sorted(words)[:15],
        })
    classes.sort(key=lambda row: (-row["size"], row["class_id"]))
    return {
        "class_count": len(classes),
        "active_class_count": sum(row["observer_weight"] > 0 for row in classes),
        "member_count": sum(row["size"] for row in classes),
        "largest_classes": classes[:12],
    }


def _calibration_rows(rows: list[dict], groups: set[str], *, max_tokens: int) -> list[dict]:
    ordered = sorted(groups, key=lambda group: hashlib.sha256(
        f"class-calibration-v2:{group}".encode()).digest())
    selected: list[dict] = []
    total = 0
    for group in ordered:
        group_rows = [row for row in rows if _group_key(row) == group]
        group_tokens = _tokens(group_rows)
        if selected and total + group_tokens > max_tokens:
            continue
        selected.extend(group_rows)
        total += group_tokens
    if not selected:
        raise ValueError("validation calibration split has no source groups")
    return selected


def _run_dataset(name: str, path: Path, *, seed: int, max_validation_tokens: int,
                 bootstrap_samples: int) -> dict:
    started = time.monotonic()
    rows = read_jsonl(path)
    split = split_documents(rows, seed)
    train_rows, validation_rows, test_rows = (split[key] for key in ("train", "validation", "test"))
    group_sets = [{_group_key(row) for row in split[key]} for key in ("train", "validation", "test")]
    if any(group_sets[left] & group_sets[right]
           for left, right in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError(f"{name}: source groups overlap across train/validation/test")

    validation_groups = sorted({_group_key(row) for row in validation_rows}, key=lambda group: hashlib.sha256(
        f"class-eval-v2:{group}".encode()).digest())
    selection_groups = set(validation_groups[::2])
    calibration_groups = set(validation_groups[1::2])
    selection_texts = [row["text"] for row in validation_rows
                       if _group_key(row) in selection_groups]
    calibration_rows = _calibration_rows(
        validation_rows, calibration_groups, max_tokens=max_validation_tokens,
    )
    calibration_texts = [row["text"] for row in calibration_rows]

    model = HierarchicalNextUnitModel(max_order=2, min_word_count=1, class_nodes=True)
    # Class geometry and global compatibility selection are validation-only;
    # the fit corpus supplies profiles and pooled count tables.
    model._class_validation_documents = selection_texts
    def report_progress(event: dict) -> None:
        if event.get("fit_stage") != "counting_documents":
            print(json.dumps({"dataset": name, "progress": event}, default=str), flush=True)
        elif event.get("completed_batches", 0) % 20 == 0:
            print(json.dumps({"dataset": name, "progress": event}, default=str), flush=True)

    model.fit([row["text"] for row in train_rows], workers=1,
              progress_callback=report_progress,
              _skip_observer_validation_calibration=True)

    # Calibrate shared word-order weights on the class-free predictor. Turning
    # off class_nodes_enabled prevents its explicit flat-count fallback from
    # bypassing the Observer while class_lambda is temporarily zero.
    original_tune = model.tune_observer_weights
    saved_class_lambda = model.class_lambda
    saved_class_enabled = model.class_nodes_enabled
    model.class_lambda = 0.0
    model.class_nodes_enabled = False
    try:
        observer_tuning = original_tune(
            calibration_texts, character=False, passes=1,
            max_events=max_validation_tokens,
        )
    finally:
        model.class_lambda = saved_class_lambda
        model.class_nodes_enabled = saved_class_enabled

    class_reinforcement = model._reinforce_class_observer_weights(calibration_texts)
    selection = model.class_selection or {}
    selected_lambdas = selection.get("selected_lambda_by_class", {})
    test_texts = [row["text"] for row in test_rows]
    comparison = model.evaluate_class_generalization(
        test_texts,
        group_ids=[_group_key(row) for row in test_rows],
        bootstrap_samples=bootstrap_samples,
    )
    overall = model.evaluate(test_texts)
    return {
        "source_file": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "context_order": 2,
        "split_rows": {key: len(value) for key, value in split.items()},
        "split_groups": {key: len(group_sets[index])
                         for index, key in enumerate(("train", "validation", "test"))},
        "split_tokens": {key: _tokens(value) for key, value in split.items()},
        "validation_group_partition": {
            "class_selection_groups": len(selection_groups),
            "observer_and_class_reinforcement_groups": len(
                {_group_key(row) for row in calibration_rows}),
            "observer_and_class_reinforcement_documents": len(calibration_rows),
            "observer_and_class_reinforcement_tokens": _tokens(calibration_rows),
            "maximum_validation_tokens": max_validation_tokens,
        },
        "class_fit_selection_summary": {
            "selected_tau": selection.get("selected_tau"),
            "maximum_selected_class_lambda": max(selected_lambdas.values(), default=0.0),
            "active_class_count": len(set(model.class_labels.values())),
            "selection_validation_gain_nats": selection.get("selected_validation_gain_nats"),
            "validation_method": selection.get("validation_method"),
        },
        "validation_observer_selection": {
            key: value for key, value in observer_tuning.items()
            if key in ("selected_weights", "validation_cross_entropy_before",
                       "validation_cross_entropy_after", "selection_split")
        },
        "validation_per_class_reinforcement": class_reinforcement,
        "test_seen_unseen_pairs": comparison,
        "test_overall": {key: overall[key] for key in
                         ("tokens", "cross_entropy_nats", "perplexity")},
        "class_quality": _class_quality(model),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def run(*, output: Path, datasets: dict[str, Path] | None = None, seed: int = 17,
        max_validation_tokens: int = 3000, bootstrap_samples: int = 1000) -> dict:
    if max_validation_tokens < 1 or bootstrap_samples < 0:
        raise ValueError("validation token and bootstrap limits must be positive/nonnegative")
    sources = datasets or DEFAULT_DATASETS
    result = {
        "protocol": (
            "source-grouped train/validation/test; class selection on one validation half; "
            "word Observer tuning and per-class reinforcement on disjoint whole validation "
            "groups, capped by tokens; max_order=2; same Observer weights for plain and "
            "class-mixed test predictors; paired source-group bootstrap"
        ),
        "seed": seed,
        "max_validation_tokens": max_validation_tokens,
        "bootstrap_samples": bootstrap_samples,
        "datasets": {},
    }
    for name, path in sources.items():
        print(json.dumps({"dataset": name, "stage": "start", "source": str(path)}), flush=True)
        result["datasets"][name] = _run_dataset(
            name, path, seed=seed, max_validation_tokens=max_validation_tokens,
            bootstrap_samples=bootstrap_samples,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"dataset": name, "stage": "complete",
                          "elapsed_seconds": result["datasets"][name]["elapsed_seconds"],
                          "unseen_pairs": result["datasets"][name]["test_seen_unseen_pairs"]["unseen_pairs"]}),
              flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "hpm_ai_v2/experiments/results/2026-10-25-class-nodes-real-text-v2/comparison.json")
    parser.add_argument("--oasst", type=Path, default=DEFAULT_DATASETS["oasst1_2500"])
    parser.add_argument("--wikitext", type=Path, default=DEFAULT_DATASETS["wikitext2_1500"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-validation-tokens", type=int, default=3000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    run(output=args.output,
        datasets={"oasst1_2500": args.oasst, "wikitext2_1500": args.wikitext},
        seed=args.seed, max_validation_tokens=args.max_validation_tokens,
        bootstrap_samples=args.bootstrap_samples)


if __name__ == "__main__":
    main()
