"""Compare the book/Python HFN checkpoint with tuned count and neural LMs.

All models fit the same 238-document source-grouped training split, select
hyperparameters on the 95-document validation split, and score the same
52-document held-out test split. Existing HFN test metrics are read from its
saved report; test documents are never used for model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from hpm_ai_v2.training.kneser_ney import KneserNeyBaseline
from hpm_ai_v2.training.language_prediction import (
    HierarchicalNextUnitModel, read_jsonl, split_documents, split_sentences, tokenize,
)
from hpm_ai_v2.training.ngram_baseline import NGramBaseline
from hpm_ai_v2.training.neural_ngram import NeuralNGramBaseline


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/pretraining/new_sources_round1.jsonl"
HFN_METRICS = ROOT / "hpm_ai_v2/experiments/results/2026-10-25-language-class-nodes-book-python-v1/metrics.json"


def _score_docs(model, docs):
    rows = []
    for index, text in enumerate(docs):
        score = model.score_cross_entropy([text])
        if isinstance(score, dict):
            tokens, cross_entropy = score["tokens"], score["cross_entropy_nats"]
        else:
            tokens = sum(len(tokenize(sentence)) + 1 for sentence in split_sentences(text))
            cross_entropy = score
        rows.append({"document": index, "tokens": tokens,
                     "cross_entropy_nats": cross_entropy})
    return rows


def _bootstrap_delta(left, right, *, seed=17, reps=5000):
    """Token-weighted paired-document CE difference, left minus right."""
    if len(left) != len(right):
        raise ValueError("paired document score vectors differ in length")
    n = len(left)
    delta = np.asarray([a["cross_entropy_nats"] - b["cross_entropy_nats"]
                        for a, b in zip(left, right)], dtype=float)
    weights = np.asarray([min(a["tokens"], b["tokens"]) for a, b in zip(left, right)], dtype=float)
    observed = float(np.average(delta, weights=weights))
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps)
    for i in range(reps):
        sample = rng.integers(0, n, size=n)
        estimates[i] = np.average(delta[sample], weights=weights[sample])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"mean_delta_nats_left_minus_right": observed,
            "bootstrap_95pct_ci": [float(low), float(high)],
            "resampling_unit": "source-grouped document", "documents": n,
            "replicates": reps}


def run(*, output: Path, neural_epochs: int = 4, neural_batch_size: int = 128):
    started = time.monotonic()
    source_rows = read_jsonl(CORPUS)
    current_partition = split_documents(source_rows, seed=17)
    checkpoint = json.loads((HFN_METRICS.parent / "hpm_language_checkpoint.json").read_text(
        encoding="utf-8"
    ))
    train_hashes = set(checkpoint["training_document_hashes"])
    test_hashes = {hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                   for row in current_partition["validation"]}
    row_hash = lambda row: hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
    if train_hashes & test_hashes:
        raise RuntimeError("saved HFN training hashes overlap the reconstructed 52-document test set")
    split = {
        "train": [row for row in source_rows if row_hash(row) in train_hashes],
        "test": [row for row in source_rows if row_hash(row) in test_hashes],
        "validation": [row for row in source_rows if row_hash(row) not in train_hashes | test_hashes],
    }
    train = [row["text"] for row in split["train"]]
    validation = [row["text"] for row in split["validation"]]
    test = [row["text"] for row in split["test"]]
    if tuple(map(len, (train, validation, test))) != (238, 95, 52):
        raise RuntimeError(f"expected the saved run split 238/95/52, got "
                           f"{len(train)}/{len(validation)}/{len(test)}")
    token_counts = {
        key: sum(len(tokenize(sentence)) + 1 for row in split[key]
                 for sentence in split_sentences(row["text"]))
        for key in ("train", "validation", "test")
    }
    if token_counts != {"train": 282631, "validation": 104860, "test": 63350}:
        raise RuntimeError(f"reconstructed split does not match saved pilot token counts: {token_counts}")
    def source_group(row):
        metadata = row.get("metadata") or {}
        source_path = row.get("source_path") or metadata.get("source_path")
        source_url = row.get("source_url") or metadata.get("source_url")
        return (f"source_path:{source_path}" if source_path else
                f"source_url:{source_url}" if source_url else
                str(row.get("group_id", row.get("id"))))
    group_sets = [{source_group(row) for row in split[key]}
                  for key in ("train", "validation", "test")]
    if any(group_sets[i] & group_sets[j]
           for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("reconstructed train/validation/test sources overlap")

    hfn_report = json.loads(HFN_METRICS.read_text(encoding="utf-8"))
    expected_hfn = hfn_report["hpm"]["test"]
    checkpoint_path = HFN_METRICS.parent / "hpm_language_checkpoint.json"
    hfn_model = HierarchicalNextUnitModel.load(checkpoint_path)
    hfn_doc_scores = _score_docs(hfn_model, test)
    hfn_recomputed = float(np.average(
        [row["cross_entropy_nats"] for row in hfn_doc_scores],
        weights=[row["tokens"] for row in hfn_doc_scores],
    ))
    # A class-enabled checkpoint that selected no classes intentionally falls
    # back to the flat count expert. Score the stored Observer weights as a
    # separate diagnostic, without presenting that counterfactual as the
    # checkpoint's actual prediction path.
    observer_weighted_model = HierarchicalNextUnitModel.load(checkpoint_path)
    observer_weighted_model.class_nodes_enabled = False
    observer_weighted_doc_scores = _score_docs(observer_weighted_model, test)
    observer_weighted_ce = float(np.average(
        [row["cross_entropy_nats"] for row in observer_weighted_doc_scores],
        weights=[row["tokens"] for row in observer_weighted_doc_scores],
    ))
    hfn_report_delta = hfn_recomputed - expected_hfn["cross_entropy_nats"]
    min_word_count = int(hfn_report["min_word_count"])
    ngram = NGramBaseline(order=3, alpha=.1, smoothing="interpolated",
                          min_word_count=min_word_count).fit(train)
    alpha_grid = (.1, 1.0, 10.0, 30.0, 100.0, 200.0, 300.0)
    ngram_val = {str(alpha): (setattr(ngram, "alpha", alpha) or ngram.score_cross_entropy(validation))
                 for alpha in alpha_grid}
    ngram_alpha = min(alpha_grid, key=lambda alpha: ngram_val[str(alpha)])
    ngram.alpha = ngram_alpha
    ngram_test = ngram.score_cross_entropy(test)
    ngram_doc_scores = _score_docs(ngram, test)

    kn_models = {}
    kn_val = {}
    kn_doc_scores = {}
    kn_unknown_priors = (0.0, 100.0, 1000.0, 5000.0, 10000.0, 20000.0)
    for discount in (.4, .55, .65, .75, .85, .95, .99):
        model = KneserNeyBaseline(order=3, discount=discount,
                                  min_word_count=min_word_count).fit(train)
        kn_models[discount] = model
        for unk_prior in kn_unknown_priors:
            model.unknown_continuation_prior = unk_prior
            validation_score = model.score_cross_entropy(validation)
            kn_val[f"discount={discount};unk_prior={unk_prior}"] = validation_score
    kn_key = min(kn_val, key=lambda key: kn_val[key]["cross_entropy_nats"])
    kn_discount = float(kn_key.split(";")[0].split("=")[1])
    kn_unknown_prior = float(kn_key.split(";")[1].split("=")[1])
    kn = kn_models[kn_discount]
    kn.unknown_continuation_prior = kn_unknown_prior
    kn_test = kn.score_cross_entropy(test)
    kn_doc_scores = _score_docs(kn, test)

    neural = NeuralNGramBaseline(context_size=2, embedding_dim=24, hidden_dim=48,
                                 min_word_count=min_word_count, seed=17,
                                 batch_size=neural_batch_size)
    neural.fit(train, validation, epochs=neural_epochs)
    if len(neural.vocab) != len(hfn_model.vocab):
        raise RuntimeError(f"neural vocabulary size {len(neural.vocab)} differs from HFN "
                           f"{len(hfn_model.vocab)}")
    neural_test = neural.score_cross_entropy(test)
    neural_doc_scores = _score_docs(neural, test)

    hfn_doc_tokens = sum(row["tokens"] for row in kn_doc_scores)
    if hfn_doc_tokens != expected_hfn["tokens"]:
        raise RuntimeError(f"test token mismatch: baselines={hfn_doc_tokens}, HFN={expected_hfn['tokens']}")
    result = {
        "protocol": "same source-grouped corpus and tokenizer; no test tuning",
        "corpus": str(CORPUS.relative_to(ROOT)),
        "seed": 17,
        "split_documents": {key: len(split[key]) for key in ("train", "validation", "test")},
        "split_sha256": {
            key: hashlib.sha256("\n".join(sorted(
                hashlib.sha256(row["text"].encode("utf-8")).hexdigest() for row in split[key]
            )).encode("utf-8")).hexdigest()
            for key in ("train", "validation", "test")
        },
        "hfn_saved_run": {"metrics_path": str(HFN_METRICS.relative_to(ROOT)),
                          "min_word_count": min_word_count,
                          "vocabulary_size": len(hfn_model.vocab),
                          "saved_test": expected_hfn,
                          "reloaded_checkpoint_test": {
                              "tokens": sum(row["tokens"] for row in hfn_doc_scores),
                              "cross_entropy_nats": hfn_recomputed,
                              "perplexity": math.exp(hfn_recomputed),
                          },
                          "reloaded_minus_saved_cross_entropy_nats": hfn_report_delta,
                          "saved_metric_reproduced_from_checkpoint": abs(hfn_report_delta) <= 1e-10,
                          "class_selection": hfn_report["class_node_training"]["selection"]},
        "hfn_observer_weighted_no_class_counterfactual": {
            "cross_entropy_nats": observer_weighted_ce,
            "perplexity": math.exp(observer_weighted_ce),
            "test_tokens": sum(row["tokens"] for row in observer_weighted_doc_scores),
            "interpretation": (
                "diagnostic only: class-node fallback is disabled so the saved per-order "
                "Observer weights are used; this is not the class-enabled checkpoint path"
            ),
        },
        "interpolated_trigram": {"selected_alpha": ngram_alpha,
                                 "validation_sweep": ngram_val,
                                 "test": ngram_test},
        "kneser_ney_trigram": {"selected_discount": kn_discount,
                               "selected_unknown_continuation_prior": kn_unknown_prior,
                               "validation_sweep": kn_val,
                               "test": kn_test},
        "neural_feedforward_trigram": {"architecture": "2-token embeddings -> tanh hidden -> softmax",
                                       "epochs_requested": neural_epochs,
                                       "selected_epoch": neural.best_epoch,
                                       "history": neural.training_history,
                                       "test": neural_test},
        "paired_document_bootstrap": {
            "hfn_minus_kneser_ney": _bootstrap_delta(hfn_doc_scores, kn_doc_scores),
            "hfn_observer_weighted_no_class_minus_kneser_ney": _bootstrap_delta(
                observer_weighted_doc_scores, kn_doc_scores),
            "hfn_minus_neural": _bootstrap_delta(hfn_doc_scores, neural_doc_scores),
            "hfn_minus_interpolated": _bootstrap_delta(hfn_doc_scores, ngram_doc_scores),
            "kneser_ney_minus_interpolated": _bootstrap_delta(kn_doc_scores, ngram_doc_scores),
            "neural_minus_interpolated": _bootstrap_delta(neural_doc_scores, ngram_doc_scores),
            "neural_minus_kneser_ney": _bootstrap_delta(neural_doc_scores, kn_doc_scores),
        },
        "limitations": [
            "The neural baseline is a small feed-forward 2-token model, not a recurrent or transformer LM.",
            "Vocabulary and rare-word cutoff match the saved HFN run; all scores include EOS targets.",
            "The saved HFN aggregate CE does not reproduce when its checkpoint is reloaded and scored on the reconstructed source-grouped test set; both values are reported, and CIs use the reloaded model predictions.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(
        "hpm_ai_v2/experiments/results/2026-10-25-language-baselines-v1/comparison.json"))
    parser.add_argument("--neural-epochs", type=int, default=4)
    parser.add_argument("--neural-batch-size", type=int, default=128)
    args = parser.parse_args()
    result = run(output=args.output, neural_epochs=args.neural_epochs,
                 neural_batch_size=args.neural_batch_size)
    print(json.dumps({"output": str(args.output), "split_documents": result["split_documents"],
                      "hfn_saved_test_ce": result["hfn_saved_run"]["saved_test"]["cross_entropy_nats"],
                      "hfn_reloaded_test_ce": result["hfn_saved_run"][
                          "reloaded_checkpoint_test"]["cross_entropy_nats"],
                      "kneser_ney_test": result["kneser_ney_trigram"]["test"],
                      "neural_test": result["neural_feedforward_trigram"]["test"]}, indent=2))


if __name__ == "__main__":
    main()
