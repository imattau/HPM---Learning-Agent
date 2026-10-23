"""Run the first supervised HPM study curriculum from a JSONL file.

Example:
    PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.experiments.run_supervised_study \
        curriculum.jsonl --output-dir /tmp/hpm-study
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import tempfile
from pathlib import Path

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.domains.ollama_embedding import OllamaEmbeddingTextDomainConfig
from hpm_ai_v2.training import (
    OllamaChatBaseline,
    OllamaRetrievalBaseline,
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
        )


def _run_with_agent(
    curriculum, sources, by_split, config, cold_dir, output_dir, ollama_model,
    retention_curriculum=None, retention_episodes=(), encoder="tfidf",
    semantic_lexical_rescue_margin=0.0,
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
        _evaluate_ollama(OllamaChatBaseline(ollama_model), sources, by_split["test"])
        if ollama_model
        else None
    )
    ollama_retrieval = (
        _evaluate_ollama(OllamaRetrievalBaseline(ollama_model), sources, by_split["test"])
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


def _evaluate_ollama(baseline, sources, episodes) -> dict:
    rows = []
    for episode in episodes:
        answer = baseline.answer(episode.question, sources)
        task_correct = (
            answer.abstained
            if episode.evidence_id is None
            else (not answer.abstained and answer.evidence_id == episode.evidence_id)
        )
        answer_f1 = (
            _token_f1(episode.expected_answer, answer.answer)
            if episode.expected_answer is not None and episode.evidence_id is not None
            else None
        )
        rows.append({
            "episode_id": episode.episode_id,
            "task_type": episode.task_type,
            "task_correct": bool(task_correct),
            "expected_evidence_id": episode.evidence_id,
            "selected_evidence_id": answer.evidence_id,
            "abstained": answer.abstained,
            "answer": answer.answer,
            "answer_token_f1": answer_f1,
            "elapsed_ms": answer.elapsed_ms,
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
            "retrieved_source_ids": list(answer.retrieved_source_ids),
        })
    answer_f1s = [row["answer_token_f1"] for row in rows if row["answer_token_f1"] is not None]
    answerable = [row for row in rows if row["expected_evidence_id"] is not None]
    latencies = [row["elapsed_ms"] for row in rows]
    task_correct = sum(row["task_correct"] for row in rows)
    evidence_correct = sum(
        row["selected_evidence_id"] == row["expected_evidence_id"] for row in answerable
    )
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
    retrieval_recall = (
        sum(row["expected_evidence_id"] in row["retrieved_source_ids"] for row in answerable)
        / len(answerable)
        if uses_retrieval and answerable else None
    )
    return {
        "model": baseline.model,
        "configuration": baseline.configuration(),
        "system": "ollama_with_bm25_retrieval" if isinstance(baseline, OllamaRetrievalBaseline) else "ollama_full_corpus",
        "runtime_model": baseline.runtime_metadata(),
        "task_accuracy": task_correct / len(rows),
        "task_accuracy_ci95": _wilson_interval(task_correct, len(rows)),
        "evidence_accuracy": (
            evidence_correct / len(answerable) if answerable else None
        ),
        "evidence_accuracy_ci95": (
            _wilson_interval(evidence_correct, len(answerable)) if answerable else None
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
    args = parser.parse_args()
    run(
        args.curriculum,
        args.output_dir,
        ollama_model=args.ollama_model,
        retention_curriculum=args.retention_curriculum,
        encoder=args.encoder,
        embedding_model=args.embedding_model,
        semantic_lexical_rescue_margin=args.semantic_lexical_rescue_margin,
    )


if __name__ == "__main__":
    main()
