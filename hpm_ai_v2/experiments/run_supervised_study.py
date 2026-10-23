"""Run the first supervised HPM study curriculum from a JSONL file.

Example:
    PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.experiments.run_supervised_study \
        curriculum.jsonl --output-dir /tmp/hpm-study
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict
from collections import Counter
from pathlib import Path

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig, tokenise
from hpm_ai_v2.domains.ollama_embedding import OllamaEmbeddingTextDomainConfig
from hpm_ai_v2.training import (
    OllamaChatBaseline,
    OllamaRetrievalBaseline,
    ModelAnswer,
    SupervisedStudyTrainer,
    load_curriculum_jsonl,
)
from hpm_ai_v2.training.study import _sentences_with_offsets, _token_f1, _wilson_interval


def run(
    curriculum: str | Path,
    output_dir: str | Path,
    ollama_model: str | None = None,
    retention_curriculum: str | Path | None = None,
    encoder: str = "tfidf",
    embedding_model: str = "qwen3-embedding:0.6b",
    semantic_lexical_rescue_margin: float = 0.0,
    baseline_history_mode: str = "relevant",
    max_prior_interactions: int = 8,
) -> dict:
    sources, episodes = load_curriculum_jsonl(curriculum)
    by_split = {
        split: [episode for episode in episodes if episode.split == split]
        for split in ("train", "validation", "test")
    }
    if not by_split["train"] or not by_split["test"]:
        raise ValueError("curriculum needs at least one train and one test episode")

    retention_sources = []
    retention_episodes = []
    if retention_curriculum is not None:
        retention_sources, retention_episodes = load_curriculum_jsonl(retention_curriculum)
        wrong_splits = [episode.episode_id for episode in retention_episodes if episode.split != "test"]
        if wrong_splits:
            raise ValueError("retention curriculum must contain only test episodes")
        if not retention_episodes:
            raise ValueError("retention curriculum needs at least one test episode")
        primary_ids = {episode.episode_id for episode in episodes}
        if primary_ids & {episode.episode_id for episode in retention_episodes}:
            raise ValueError("retention and primary curricula must use disjoint episode IDs")

    all_sources = {source.source_id: source for source in sources}
    for source in retention_sources:
        if source.source_id in all_sources and all_sources[source.source_id] != source:
            raise ValueError(f"source_id reused with different content: {source.source_id}")
        all_sources[source.source_id] = source
    sources = list(all_sources.values())

    if encoder == "tfidf":
        config = TextDomainConfig.from_passages(
            [source.text for source in sources], max_vocab=20_000
        )
    elif encoder == "ollama_embedding":
        config = OllamaEmbeddingTextDomainConfig(model=embedding_model)
        # The encoder is frozen: batch precomputation changes neither its
        # parameters nor its vocabulary and avoids one HTTP request per span.
        encoder_texts = [source.text for source in sources]
        encoder_texts.extend(
            sentence
            for source in sources
            for sentence, _start, _end in _sentences_with_offsets(source.text)
        )
        encoder_texts.extend(episode.question for episode in episodes)
        encoder_texts.extend(episode.question for episode in retention_episodes)
        config.precompute(encoder_texts)
    else:
        raise ValueError("encoder must be 'tfidf' or 'ollama_embedding'")
    # Keep this run isolated; a prior forest must not contaminate the benchmark.
    # The trained checkpoint is saved separately under output_dir.
    with tempfile.TemporaryDirectory(prefix="hpm_supervised_study_") as cold_dir:
        return _run_with_agent(
            curriculum, sources, by_split, config, cold_dir, output_dir, ollama_model,
            retention_curriculum, retention_episodes,
            encoder,
            semantic_lexical_rescue_margin,
            baseline_history_mode, max_prior_interactions,
        )


def _run_with_agent(
    curriculum, sources, by_split, config, cold_dir, output_dir, ollama_model,
    retention_curriculum=None, retention_episodes=(), encoder="tfidf",
    semantic_lexical_rescue_margin=0.0,
    baseline_history_mode="relevant", max_prior_interactions=8,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    event_log = Path(output_dir) / "training_events.jsonl"
    event_log.write_text("", encoding="utf-8")
    agent = ReaderAgent(
        config,
        dynamic_vocab=False,
        inject_blank_priors=False,
        cold_dir=cold_dir,
    )
    trainer = SupervisedStudyTrainer(
        agent,
        event_log=event_log,
        semantic_lexical_rescue_margin=semantic_lexical_rescue_margin,
    )

    # Baseline uses the same fixed text representation with neutral HFN weights.
    trainer.ingest(sources)
    baseline = _evaluate_lexical(agent, trainer, by_split["test"])
    prior_interactions = _supervised_demonstrations(
        [*by_split["train"], *by_split["validation"]]
    )

    # Re-run fit with the already-ingested data; this applies only training
    # episodes and uses validation strictly for abstention calibration.
    if retention_episodes:
        retention_result = trainer.fit_phase(
            [], by_split["train"], by_split["validation"],
            retention_episodes=retention_episodes,
        )
        fit_summary = retention_result["fit"]
        retention = {
            "curriculum": str(retention_curriculum),
            "before": retention_result["retention_before"],
            "after": retention_result["retention_after"],
            "accuracy_delta": retention_result["retention_accuracy_delta"],
            "max_accuracy_drop": retention_result["retention_max_accuracy_drop"],
            "gate_passed": retention_result["retention_gate_passed"],
        }
    else:
        fit_summary = trainer.fit([], by_split["train"], by_split["validation"])
        retention = None
    hpm = trainer.evaluate(by_split["test"])
    ollama = (
        _evaluate_ollama(
            OllamaChatBaseline(ollama_model), sources, by_split["test"],
            prior_interactions=prior_interactions,
            history_mode=baseline_history_mode,
            max_prior_interactions=max_prior_interactions,
            progress_path=Path(output_dir) / "ollama_baseline_progress.jsonl",
        )
        if ollama_model
        else None
    )
    ollama_retrieval = (
        _evaluate_ollama(
            OllamaRetrievalBaseline(ollama_model), sources, by_split["test"],
            prior_interactions=prior_interactions,
            history_mode=baseline_history_mode,
            max_prior_interactions=max_prior_interactions,
            progress_path=Path(output_dir) / "ollama_retrieval_progress.jsonl",
        )
        if ollama_model
        else None
    )
    checkpoint = trainer.save_checkpoint(output_dir)
    restored_agent = ReaderAgent.load_agent(str(output_dir))
    restored_trainer = SupervisedStudyTrainer(restored_agent)
    restored_trainer.load_checkpoint_state(output_dir)
    restart_hpm = restored_trainer.evaluate(by_split["test"])
    prediction_signature = lambda metrics: [
        (
            row["episode_id"],
            row["selected_evidence_id"],
            row["answer"],
            row["citations"],
        )
        for row in metrics["episode_results"]
    ]
    restart_consistency = {
        "same_predictions": prediction_signature(hpm) == prediction_signature(restart_hpm),
        "pre_restart_accuracy": hpm["accuracy"],
        "post_restart_accuracy": restart_hpm["accuracy"],
        "post_restart_metrics": restart_hpm,
    }
    report = {
        "curriculum": str(curriculum),
        "host": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "processor": _processor_name(),
            "logical_cpu_count": os.cpu_count(),
            "gpus": _gpu_metadata(),
        },
        "hpm_encoder": (
            {
                **config.encoder_metadata(),
                "runtime_model": OllamaChatBaseline(config.embedding_model).runtime_metadata(),
            }
            if encoder == "ollama_embedding" else {"type": "tfidf"}
        ),
        "hpm_retrieval_policy": {
            "learned_fact_bonus": 0.1,
            "learned_concept_bonus_by_status": {
                "candidate": 0.0,
                "validated": 0.05,
                "promoted": 0.08,
            },
            "validated_fact_status_factors": {
                "candidate": 0.95,
                "validated": 1.0,
                "promoted": 1.05,
            },
            "semantic_bm25_bonus": trainer.semantic_lexical_bonus,
            "semantic_bm25_normalization": "score / (score + 2.0)",
            "semantic_lexical_rescue_threshold": trainer.semantic_lexical_rescue_threshold,
            "semantic_lexical_rescue_margin": trainer.semantic_lexical_rescue_margin,
            "abstention_similarity_threshold": trainer.abstention_similarity_threshold,
            "fact_retrieval_similarity_threshold": trainer.fact_retrieval_similarity_threshold,
            "concept_retrieval_similarity_threshold": trainer.concept_retrieval_similarity_threshold,
            "trained_abstention_patterns": len(trainer.abstention_node_data),
        },
        "source_count": len(sources),
        "train_count": len(by_split["train"]),
        "validation_count": len(by_split["validation"]),
        "test_count": len(by_split["test"]),
        "fit": fit_summary,
        "retention": retention,
        "baseline_cosine": baseline,
        "hpm_supervised": hpm,
        "restart_consistency": restart_consistency,
        "ollama_baseline": ollama,
        "ollama_retrieval_baseline": ollama_retrieval,
        "baseline_history_policy": {
            "mode": baseline_history_mode,
            "max_prior_interactions": max_prior_interactions if baseline_history_mode == "relevant" else None,
        },
        "checkpoint": str(checkpoint),
    }
    report_path = Path(output_dir) / "study_report.json"
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def _evaluate_lexical(agent, trainer, episodes) -> dict:
    eligible = [episode for episode in episodes if episode.evidence_id is not None]
    if not eligible:
        return {"evidence_accuracy": None, "evidence_accuracy_ci95": None, "correct": 0, "total": 0}
    correct = 0
    for episode in eligible:
        ranked = agent.query_scored_nodes(episode.question, k=len(trainer.sources))
        ranked_ids = [trainer.node_to_source[node_id] for node_id, _, _ in ranked
                      if node_id in trainer.node_to_source]
        correct += bool(ranked_ids and ranked_ids[0] == episode.evidence_id)
    return {
        "evidence_accuracy": correct / len(eligible),
        "evidence_accuracy_ci95": _wilson_interval(correct, len(eligible)),
        "correct": correct,
        "total": len(eligible),
    }


def _processor_name() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor().strip() or None


def _gpu_metadata() -> dict:
    """Capture host GPU names and driver details when nvidia-smi is available."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "unavailable", "reason": "nvidia-smi_not_found"}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "reason": type(exc).__name__}
    if result.returncode != 0:
        return {"status": "unavailable", "reason": "nvidia_smi_failed"}
    devices = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 3:
            continue
        devices.append({
            "name": row[0].strip(),
            "memory_total": row[1].strip(),
            "driver_version": row[2].strip(),
        })
    return {"status": "available" if devices else "unavailable", "devices": devices}


def _supervised_demonstrations(episodes) -> list[dict]:
    """Serialize only pre-test labeled interactions for the matched baselines."""
    demonstrations = []
    for episode in episodes:
        if episode.split == "test":
            continue
        outcome = {
            "abstained": episode.evidence_id is None and episode.task_type not in {"preference", "procedure", "compare"},
            "evidence_id": episode.evidence_id,
            "answer": episode.expected_answer,
        }
        if episode.evidence_ids:
            outcome["evidence_ids"] = list(episode.evidence_ids)
        if episode.workflow_key is not None:
            outcome["workflow_key"] = episode.workflow_key
        demonstrations.append({
            "episode_id": episode.episode_id,
            "user_id": episode.user_id,
            "task_type": episode.task_type,
            "preference_key": episode.preference_key,
            "concept_key": episode.concept_key,
            "workflow_key": episode.workflow_key,
            "user_request": episode.question,
            "outcome": outcome,
            "correction_of": episode.correction_of,
        })
    return demonstrations


def _select_relevant_interactions(
    question: str,
    interactions: list[dict],
    *,
    task_type: str,
    preference_key: str | None,
    limit: int,
) -> list[dict]:
    """Retrieve a bounded, query-relevant few-shot set from the user history."""
    if task_type == "preference":
        matching = [
            row for row in interactions
            if row.get("task_type") == "preference"
            and row.get("preference_key") == preference_key
        ]
        return matching[-limit:]
    candidates = [
        (index, row) for index, row in enumerate(interactions)
        if row.get("task_type") != "preference"
    ]
    query_terms = set(tokenise(question))
    if not query_terms or not candidates:
        return []
    token_rows = []
    for _index, row in candidates:
        outcome = row.get("outcome") or {}
        text = " ".join((
            str(row.get("user_request", "")),
            str(outcome.get("answer", "")),
            str(row.get("workflow_key", outcome.get("workflow_key", ""))),
        ))
        token_rows.append(tokenise(text))
    df = {term: sum(term in tokens for tokens in token_rows) for term in query_terms}
    average_length = sum(map(len, token_rows)) / max(len(token_rows), 1) or 1.0
    k1, b = 1.5, 0.75
    scored = []
    for (index, row), tokens in zip(candidates, token_rows):
        frequencies = Counter(tokens)
        length = len(tokens)
        score = 0.0
        for term in query_terms:
            tf = frequencies.get(term, 0)
            if not tf:
                continue
            idf = math.log1p((len(token_rows) - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * length / average_length))
        if score > 0:
            scored.append((score, index, row))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return [row for _score, _index, row in scored[:limit]]


def _evaluate_ollama(
    baseline, sources, episodes, *, prior_interactions=(),
    history_mode: str = "all", max_prior_interactions: int = 8,
    progress_path: str | Path | None = None,
) -> dict:
    sources = list(sources)
    episodes = list(episodes)
    prior_interactions = list(prior_interactions)
    if history_mode not in {"all", "relevant"}:
        raise ValueError("history_mode must be 'all' or 'relevant'")
    if max_prior_interactions < 1:
        raise ValueError("max_prior_interactions must be positive")
    progress_file = Path(progress_path) if progress_path is not None else None
    completed: dict[str, dict] = {}
    if progress_file is not None:
        signature_payload = {
            "baseline": baseline.configuration(),
            # Provenance annotations do not alter baseline retrieval or its
            # prompt, so keep them out of the resumable scoring signature.
            "sources": [
                {"source_id": source.source_id, "text": source.text, "title": source.title}
                for source in sources
            ],
            "episodes": [asdict(episode) for episode in episodes],
            "prior_interactions": list(prior_interactions),
            "history_mode": history_mode,
            "max_prior_interactions": max_prior_interactions,
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        if progress_file.exists():
            with progress_file.open(encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            if records:
                if records[0].get("_meta", {}).get("signature") != signature:
                    raise ValueError(f"Ollama progress file does not match this evaluation: {progress_file}")
                completed = {
                    row["episode_id"]: row for row in records[1:]
                    if "episode_id" in row
                }
            else:
                with progress_file.open("w", encoding="utf-8") as stream:
                    stream.write(json.dumps({"_meta": {"signature": signature}}) + "\n")
                    stream.flush()
        else:
            with progress_file.open("w", encoding="utf-8") as stream:
                stream.write(json.dumps({"_meta": {"signature": signature}}) + "\n")
                stream.flush()
    rows = []
    available_history_counts = {}
    selected_history_counts = {}
    for episode in episodes:
        available_history = [
            row for row in prior_interactions
            if row.get("user_id", "default") == episode.user_id
        ]
        available_history_counts[episode.user_id] = len(available_history)
        user_history = (
            _select_relevant_interactions(
                episode.question, available_history, task_type=episode.task_type,
                preference_key=episode.preference_key, limit=max_prior_interactions,
            )
            if history_mode == "relevant" else available_history
        )
        selected_history_counts[episode.user_id] = len(user_history)
        if episode.episode_id in completed:
            rows.append(completed[episode.episode_id])
            continue
        answer_error = None
        started = time.perf_counter()
        try:
            answer = baseline.answer(
                episode.question,
                sources,
                prior_interactions=user_history,
                task_type=episode.task_type,
                user_id=episode.user_id,
                preference_key=episode.preference_key,
            )
        except Exception as exc:
            answer_error = f"{type(exc).__name__}: {str(exc)[:240]}"
            answer = ModelAnswer(
                answer="", evidence_id=None, abstained=False,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        task_correct = (
            (
                not answer.abstained
                and _token_f1(episode.expected_answer or "", answer.answer) >= 0.8
            )
            if episode.task_type == "preference"
            else set(answer.evidence_ids) == set(episode.evidence_ids)
            if episode.task_type == "compare"
            else answer.abstained
            if episode.evidence_id is None
            else (not answer.abstained and answer.evidence_id == episode.evidence_id)
        )
        if episode.task_type == "procedure":
            task_correct = answer.workflow_key == episode.workflow_key
        if answer_error is not None:
            task_correct = False
        answer_f1 = (
            _token_f1(episode.expected_answer, answer.answer)
            if episode.expected_answer is not None
            and (episode.evidence_id is not None or episode.task_type == "preference")
            else None
        )
        row = {
            "episode_id": episode.episode_id,
            "task_type": episode.task_type,
            "expected_workflow_key": episode.workflow_key,
            "selected_workflow_key": answer.workflow_key,
            "task_correct": bool(task_correct),
            "expected_evidence_id": episode.evidence_id,
            "selected_evidence_id": answer.evidence_id,
            "expected_evidence_ids": list(episode.evidence_ids),
            "selected_evidence_ids": list(answer.evidence_ids),
            "abstained": answer.abstained,
            "answer": answer.answer,
            "answer_token_f1": answer_f1,
            "evaluation_error": answer_error,
            "elapsed_ms": answer.elapsed_ms,
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
            "retrieved_source_ids": list(answer.retrieved_source_ids),
            "prior_interaction_count": len(user_history),
            "available_prior_interaction_count": len(available_history),
        }
        rows.append(row)
        if progress_file is not None:
            with progress_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
    answer_f1s = [row["answer_token_f1"] for row in rows if row["answer_token_f1"] is not None]
    answerable = [row for row in rows if row["expected_evidence_id"] is not None]
    compare_rows = [row for row in rows if row["task_type"] == "compare"]
    comparison_expected = sum(len(row["expected_evidence_ids"]) for row in compare_rows)
    comparison_correct = sum(
        len(set(row["expected_evidence_ids"]) & set(row["selected_evidence_ids"]))
        for row in compare_rows
    )
    comparison_selected = sum(len(row["selected_evidence_ids"]) for row in compare_rows)
    latencies = [row["elapsed_ms"] for row in rows]
    task_correct = sum(row["task_correct"] for row in rows)
    evidence_correct = sum(
        row["selected_evidence_id"] == row["expected_evidence_id"] for row in answerable
    ) + comparison_correct
    evidence_total = len(answerable) + comparison_expected
    per_task_type = {}
    for row in rows:
        bucket = per_task_type.setdefault(row["task_type"], {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int(row["task_correct"])
    for task, bucket in per_task_type.items():
        task_f1s = [
            row["answer_token_f1"] for row in rows
            if row["task_type"] == task and row["answer_token_f1"] is not None
        ]
        bucket["answer_reference_count"] = len(task_f1s)
        bucket["answer_token_f1"] = sum(task_f1s) / len(task_f1s) if task_f1s else None
    uses_retrieval = isinstance(baseline, OllamaRetrievalBaseline)
    retrieved_rows = [row for row in rows if row["retrieved_source_ids"]]
    comparison_links = [
        (expected_id, row)
        for row in compare_rows
        for expected_id in row["expected_evidence_ids"]
    ]
    comparison_retrieved = sum(
        expected_id in row["retrieved_source_ids"]
        for expected_id, row in comparison_links
    )
    retrieval_hits = sum(
        row["expected_evidence_id"] in row["retrieved_source_ids"]
        for row in answerable
    ) + comparison_retrieved
    retrieval_denominator = len(answerable) + len(comparison_links)
    grounded_answer_rows = [
        row for row in answerable
        if row["answer_token_f1"] is not None
    ]
    grounded_answer_correct = sum(
        not row["abstained"]
        and row["selected_evidence_id"] == row["expected_evidence_id"]
        and row["answer_token_f1"] >= 0.8
        and row["evaluation_error"] is None
        for row in grounded_answer_rows
    )
    retrieval_recall = (
        retrieval_hits / retrieval_denominator
        if uses_retrieval and retrieval_denominator else None
    )
    configuration = baseline.configuration()
    history_counts = {row["prior_interaction_count"] for row in rows}
    uniform_history_count = next(iter(history_counts)) if len(history_counts) == 1 else None
    configuration["prior_interaction_count"] = uniform_history_count
    configuration["prior_interaction_count_by_user"] = selected_history_counts
    configuration["available_prior_interaction_count_by_user"] = available_history_counts
    configuration["history_mode"] = history_mode
    configuration["max_prior_interactions"] = max_prior_interactions if history_mode == "relevant" else None
    return {
        "model": baseline.model,
        "configuration": configuration,
        "prior_interaction_count": uniform_history_count,
        "prior_interaction_count_by_user": selected_history_counts,
        "available_prior_interaction_count_by_user": available_history_counts,
        "history_mode": history_mode,
        "system": "ollama_with_bm25_retrieval" if isinstance(baseline, OllamaRetrievalBaseline) else "ollama_full_corpus",
        "runtime_model": baseline.runtime_metadata(),
        "task_accuracy": task_correct / len(rows),
        "task_accuracy_ci95": _wilson_interval(task_correct, len(rows)),
        "evidence_accuracy": (
            evidence_correct / evidence_total if evidence_total else None
        ),
        "evidence_accuracy_ci95": (
            _wilson_interval(evidence_correct, evidence_total) if evidence_total else None
        ),
        "grounded_answer_accuracy": (
            grounded_answer_correct / len(grounded_answer_rows)
            if grounded_answer_rows else None
        ),
        "grounded_answer_correct": grounded_answer_correct,
        "grounded_answer_total": len(grounded_answer_rows),
        "grounded_answer_f1_threshold": 0.8,
        "comparison_evidence_precision": (
            comparison_correct / comparison_selected if comparison_selected else None
        ),
        "comparison_evidence_recall": (
            comparison_correct / comparison_expected if comparison_expected else None
        ),
        "answer_token_f1": sum(answer_f1s) / len(answer_f1s) if answer_f1s else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "retrieval_diagnostics": ({
            "answerable_source_recall_at_k": retrieval_recall,
            "answerable_total": len(answerable),
            "mean_retrieved_sources_per_episode": (
                sum(len(row["retrieved_source_ids"]) for row in rows) / len(rows)
            ),
            "episodes_with_retrieval": len(retrieved_rows),
        } if uses_retrieval else None),
        "per_task_type": {
            task: {
                **counts,
                "accuracy": counts["correct"] / counts["total"],
                "accuracy_ci95": _wilson_interval(counts["correct"], counts["total"]),
            }
            for task, counts in sorted(per_task_type.items())
        },
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("curriculum", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ollama-model",
        help="also evaluate a locally installed Ollama model on the same held-out episodes",
    )
    parser.add_argument(
        "--retention-curriculum",
        type=Path,
        help="optional JSONL curriculum containing a fixed, separate test suite for pre/post-phase retention scoring",
    )
    parser.add_argument(
        "--encoder",
        choices=("tfidf", "ollama_embedding"),
        default="tfidf",
        help="fixed HPM text representation to evaluate",
    )
    parser.add_argument(
        "--embedding-model",
        default="qwen3-embedding:0.6b",
        help="Ollama embedding model used when --encoder=ollama_embedding",
    )
    parser.add_argument(
        "--semantic-lexical-rescue-margin",
        type=float,
        default=0.0,
        help="experimental distance below the calibrated threshold where strong BM25 support may rescue; 0 disables rescue",
    )
    parser.add_argument(
        "--baseline-history-mode",
        choices=("all", "relevant"),
        default="relevant",
        help="retrieve bounded relevant examples by default, or pass the entire user history with 'all'",
    )
    parser.add_argument(
        "--max-prior-interactions",
        type=int,
        default=8,
        help="maximum retrieved examples per test question when baseline history mode is relevant",
    )
    args = parser.parse_args()
    run(
        args.curriculum,
        args.output_dir,
        ollama_model=args.ollama_model,
        retention_curriculum=args.retention_curriculum,
        encoder=args.encoder,
        embedding_model=args.embedding_model,
        semantic_lexical_rescue_margin=args.semantic_lexical_rescue_margin,
        baseline_history_mode=args.baseline_history_mode,
        max_prior_interactions=args.max_prior_interactions,
    )


if __name__ == "__main__":
    main()
