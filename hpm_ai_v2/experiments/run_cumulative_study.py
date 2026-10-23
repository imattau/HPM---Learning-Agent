"""Train HPM over ordered study sessions and audit learning retention.

Each ``--phase`` is a JSONL curriculum with train, optional validation, and
optional test episodes. A phase's held-out test suite is evaluated after that
phase and is reused as a retention exam in later phases. An optional separate
``--retention`` curriculum supplies additional fixed test episodes.

Example::

    PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.experiments.run_cumulative_study \
        --phase session-1.jsonl --phase session-2.jsonl \
        --retention retained-exam.jsonl --output-dir /tmp/hpm-cumulative
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from contextlib import ExitStack

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.domains.ollama_embedding import OllamaEmbeddingTextDomainConfig
from hpm_ai_v2.domains.text_domain import TextDomainConfig, tokenise
from hpm_ai_v2.experiments.run_supervised_study import (
    _evaluate_ollama,
    _gpu_metadata,
    _processor_name,
    _supervised_demonstrations,
)
from hpm_ai_v2.training import (
    OllamaChatBaseline,
    OllamaRetrievalBaseline,
    SourceDocument,
    SupervisedStudyTrainer,
    load_curriculum_jsonl,
)


def _prediction_signature(metrics: dict) -> str:
    """Serialize predictions canonically so tuples and JSON arrays compare equally."""
    rows = metrics["episode_results"]
    signature = [
        (
            row["episode_id"],
            row["selected_evidence_id"],
            row["answer"],
            row["citations"],
            row.get("selected_evidence_ids", []),
            row.get("selected_workflow_key"),
            row.get("abstained"),
        )
        for row in rows
    ]
    return json.dumps(signature, sort_keys=True, ensure_ascii=False)


def run_cumulative(
    phase_paths: list[str | Path],
    retention_path: str | Path | None,
    output_dir: str | Path,
    *,
    encoder: str = "tfidf",
    embedding_model: str = "qwen3-embedding:0.6b",
    ollama_model: str | None = None,
    max_accuracy_drop: float = 0.05,
    fact_retrieval_similarity_threshold: float = 0.30,
    concept_retrieval_similarity_threshold: float = 0.30,
    expand_vocabulary: bool = True,
    correction_query_similarity_threshold: float | None = None,
    correction_answer_similarity_threshold: float | None = None,
    correction_transfer_score_threshold: float | None = None,
    baseline_history_mode: str = "relevant",
    max_prior_interactions: int = 8,
    baseline_mode: str = "both",
    initial_checkpoint: str | Path | None = None,
    baseline_history_paths: list[str | Path] | None = None,
    pretraining_corpus_paths: list[str | Path] | None = None,
) -> dict:
    if not phase_paths:
        raise ValueError("at least one --phase curriculum is required")
    if baseline_mode not in {"both", "chat", "retrieval"}:
        raise ValueError("baseline_mode must be 'both', 'chat', or 'retrieval'")
    phases = []
    phase_episodes = []
    all_sources: dict[str, SourceDocument] = {}
    seen_episode_ids: set[str] = set()
    initial_fit_ids: set[str] = set()
    initial_source_ids: set[str] = set()
    if initial_checkpoint is not None:
        checkpoint_dir = Path(initial_checkpoint)
        checkpoint_state = json.loads(
            (checkpoint_dir / "study_training_state.json").read_text(encoding="utf-8")
        )
        initial_fit_ids = set(checkpoint_state.get("fit_episode_ids", ()))
        checkpoint_sources = [SourceDocument(**row) for row in checkpoint_state.get("sources", ())]
        initial_source_ids = {source.source_id for source in checkpoint_sources}
        _merge_sources(all_sources, checkpoint_sources)

    history_by_id: dict[str, object] = {}
    for history_path in baseline_history_paths or ():
        _history_sources, history_episodes_in_file = load_curriculum_jsonl(history_path)
        for episode in history_episodes_in_file:
            if episode.split not in {"train", "validation"}:
                continue
            previous = history_by_id.get(episode.episode_id)
            if previous is not None and previous != episode:
                raise ValueError(f"conflicting baseline history episode: {episode.episode_id}")
            history_by_id[episode.episode_id] = episode
    history_episodes = list(history_by_id.values())
    if initial_checkpoint is not None and initial_fit_ids and not baseline_history_paths:
        raise ValueError(
            "checkpoint continuation requires --baseline-history-curriculum for every "
            "train/validation episode so the original validation calibration is preserved"
        )
    if initial_checkpoint is not None and baseline_history_paths:
        missing_history = initial_fit_ids - set(history_by_id)
        unexpected_history = set(history_by_id) - initial_fit_ids
        if missing_history or unexpected_history:
            raise ValueError(
                "baseline history must reproduce the checkpoint's train/validation IDs "
                f"(missing={len(missing_history)}, unexpected={len(unexpected_history)})"
            )
    for index, phase_path in enumerate(phase_paths, start=1):
        sources, episodes = load_curriculum_jsonl(phase_path)
        if not any(ep.split == "train" for ep in episodes):
            raise ValueError(f"phase {phase_path} requires train episodes")
        _merge_sources(all_sources, sources)
        duplicates = seen_episode_ids & {ep.episode_id for ep in episodes}
        if duplicates:
            raise ValueError(f"episode IDs reused between phases: {sorted(duplicates)[:3]}")
        seen_episode_ids.update(ep.episode_id for ep in episodes)
        unavailable_test_evidence = sorted({
            ep.evidence_id for ep in episodes
            if ep.split == "test" and ep.evidence_id and ep.evidence_id not in all_sources
        })
        if unavailable_test_evidence:
            raise ValueError(
                f"phase {phase_path} test episodes reference sources not yet introduced: "
                + ", ".join(unavailable_test_evidence[:5])
            )
        phase_episodes.extend(episodes)
        phases.append({
            "name": Path(phase_path).stem or f"phase-{index}",
            "sources": sources,
            "train_episodes": [ep for ep in episodes if ep.split == "train"],
            "validation_episodes": [ep for ep in episodes if ep.split == "validation"],
            "test_episodes": [ep for ep in episodes if ep.split == "test"],
        })

    # Source-only corpora seed HFN before the first held-out exam. They add
    # evidence and encoder vocabulary without inventing supervised labels.
    pretraining_sources: dict[str, SourceDocument] = {}
    for corpus_path in pretraining_corpus_paths or ():
        corpus_sources, corpus_episodes = load_curriculum_jsonl(corpus_path)
        if corpus_episodes:
            raise ValueError(
                f"pretraining corpus {corpus_path} must contain source rows only"
            )
        _merge_sources(pretraining_sources, corpus_sources)
    _merge_sources(all_sources, list(pretraining_sources.values()))
    if pretraining_sources:
        first_phase_sources = {
            source.source_id: source for source in phases[0]["sources"]
        }
        _merge_sources(first_phase_sources, list(pretraining_sources.values()))
        phases[0]["sources"] = list(first_phase_sources.values())

    if retention_path is not None:
        retention_sources, retention_episodes = load_curriculum_jsonl(retention_path)
        if not retention_episodes or any(ep.split != "test" for ep in retention_episodes):
            raise ValueError("retention curriculum must contain only test episodes")
    else:
        retention_sources, retention_episodes = [], []
    duplicates = seen_episode_ids & {ep.episode_id for ep in retention_episodes}
    if duplicates:
        raise ValueError(f"retention episode IDs overlap training episodes: {sorted(duplicates)[:3]}")
    _merge_sources(all_sources, retention_sources)
    # fit_curriculum scores retention before the first phase as well as after
    # each phase, so its evidence must already be in the initial HPM state.
    phase_source_ids = {source.source_id for source in phases[0]["sources"]}
    available_initial_source_ids = phase_source_ids | initial_source_ids
    missing = sorted(
        {ep.evidence_id for ep in retention_episodes if ep.evidence_id}
        - available_initial_source_ids
    )
    if retention_episodes and missing:
        raise ValueError(
            "retention evidence must be present before phase one; missing sources: "
            + ", ".join(missing[:5])
        )

    source_list = list(all_sources.values())
    if initial_checkpoint is not None:
        reader_meta = json.loads((Path(initial_checkpoint) / "reader_meta.json").read_text(encoding="utf-8"))
        saved_encoder = reader_meta.get("encoder", {}).get("type", "tfidf")
        requested_encoder = "ollama_embedding" if encoder == "ollama_embedding" else "tfidf"
        if requested_encoder != saved_encoder:
            raise ValueError(
                f"checkpoint uses encoder {saved_encoder!r}; requested {requested_encoder!r}"
            )
        config = None
    elif encoder == "tfidf":
        config = TextDomainConfig.from_passages(
            [source.text for source in source_list], max_vocab=20_000
        )
    elif encoder == "ollama_embedding":
        config = OllamaEmbeddingTextDomainConfig(model=embedding_model)
        texts = [source.text for source in source_list]
        from hpm_ai_v2.training.study import _sentences_with_offsets
        texts.extend(
            sentence for source in source_list
            for sentence, _, _ in _sentences_with_offsets(source.text)
        )
        texts.extend(ep.question for ep in [*phase_episodes, *retention_episodes])
        config.precompute(texts)
    else:
        raise ValueError("encoder must be 'tfidf' or 'ollama_embedding'")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    event_log = out / "training_events.jsonl"
    event_log.write_text("", encoding="utf-8")
    with ExitStack() as stack:
        vocabulary_expansion = {
            "new_source_count": 0,
            "new_terms_considered": 0,
            "terms_added": 0,
        }
        if initial_checkpoint is not None:
            # ReaderAgent.load_agent uses its directory as the TieredForest
            # cold store, and reindexing can rewrite cold node files. Continue
            # from an isolated copy so a training run never mutates its source
            # checkpoint or changes the starting point for later experiments.
            checkpoint_work_root = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="hpm_checkpoint_copy_"))
            )
            checkpoint_working_copy = checkpoint_work_root / "checkpoint"
            shutil.copytree(initial_checkpoint, checkpoint_working_copy)
            agent = ReaderAgent.load_agent(str(checkpoint_working_copy))
            if isinstance(agent.config, OllamaEmbeddingTextDomainConfig):
                from hpm_ai_v2.training.study import _sentences_with_offsets
                semantic_texts = [source.text for source in all_sources.values()]
                semantic_texts.extend(
                    sentence for source in all_sources.values()
                    for sentence, _, _ in _sentences_with_offsets(source.text)
                )
                semantic_texts.extend(
                    episode.question
                    for episode in [*phase_episodes, *retention_episodes, *history_episodes]
                )
                agent.config.precompute(semantic_texts)
                vocabulary_expansion = {
                    "new_source_count": sum(
                        source.source_id not in initial_source_ids for source in source_list
                    ),
                    "new_terms_considered": 0,
                    "terms_added": 0,
                    "enabled": False,
                    "reason": "fixed semantic embedding dimensions; new evidence was precomputed",
                }
            absent_initial_texts = [
                source.source_id for source in source_list
                if source.source_id in initial_source_ids
                and source.text not in agent.config._passages
            ]
            if absent_initial_texts:
                raise ValueError(
                    "checkpoint encoder is missing previously learned source texts: "
                    + ", ".join(absent_initial_texts[:5])
                )
            new_sources = [
                source for source in source_list
                if source.source_id not in initial_source_ids
            ]
            if not isinstance(agent.config, OllamaEmbeddingTextDomainConfig):
                # Grow a checkpoint's TF-IDF vocabulary from new source text
                # and supervised train/validation requests only.
                expansion_texts = [source.text for source in new_sources]
                expansion_texts.extend(
                    episode.question for episode in phase_episodes
                    if episode.split in {"train", "validation"}
                )
                existing_terms = set(agent.config.concepts)
                new_terms = {
                    term for text in expansion_texts for term in tokenise(text)
                    if term not in existing_terms
                }
                vocabulary_expansion = {
                    "new_source_count": len(new_sources),
                    "new_terms_considered": len(new_terms),
                    "terms_added": 0,
                    "enabled": expand_vocabulary,
                }
                if new_terms and expand_vocabulary:
                    added = agent.expand_vocabulary(
                        expansion_texts, max_new=len(new_terms)
                    )
                    vocabulary_expansion["terms_added"] = added
                    if added != len(new_terms):
                        raise RuntimeError(
                            "checkpoint vocabulary expansion did not add every term "
                            f"from new sources and supervised requests ({added}/{len(new_terms)})"
                        )
        else:
            cold_dir = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="hpm_cumulative_study_")
            )
            agent = ReaderAgent(
                config, dynamic_vocab=False, inject_blank_priors=False, cold_dir=cold_dir
            )
        trainer = SupervisedStudyTrainer(
            agent,
            event_log=event_log,
            fact_retrieval_similarity_threshold=fact_retrieval_similarity_threshold,
            concept_retrieval_similarity_threshold=concept_retrieval_similarity_threshold,
        )
        if initial_checkpoint is not None:
            trainer.load_checkpoint_state(checkpoint_working_copy)
        # Explicit overrides let a continuation recalibrate persisted
        # correction thresholds instead of silently inheriting old values.
        if correction_query_similarity_threshold is not None:
            trainer.correction_query_similarity_threshold = correction_query_similarity_threshold
        if correction_answer_similarity_threshold is not None:
            trainer.correction_answer_similarity_threshold = correction_answer_similarity_threshold
        if correction_transfer_score_threshold is not None:
            trainer.correction_transfer_score_threshold = correction_transfer_score_threshold
        training = trainer.fit_curriculum(
            phases,
            retention_episodes=retention_episodes,
            initial_validation_episodes=(
                [episode for episode in history_episodes if episode.split == "validation"]
                if initial_checkpoint is not None else ()
            ),
            max_accuracy_drop=max_accuracy_drop,
        )
        phase_test_episodes = [ep for phase in phases for ep in phase["test_episodes"]]
        final_exam = [*retention_episodes, *phase_test_episodes]
        if not final_exam:
            raise ValueError("provide --retention or include test episodes in at least one phase")
        before_restart = trainer.evaluate(final_exam)
        checkpoint = trainer.save_checkpoint(out)
        restored_agent = ReaderAgent.load_agent(str(out))
        restored = SupervisedStudyTrainer(restored_agent)
        restored.load_checkpoint_state(out)
        after_restart = restored.evaluate(final_exam)
        result = {
            "training": training,
            "retention_after_all_phases": before_restart,
            "restart_consistency": {
                "same_predictions": _prediction_signature(before_restart)
                == _prediction_signature(after_restart),
                "post_restart": after_restart,
            },
            "knowledge_growth": {
                "fact_nodes": len(trainer.fact_node_data),
                "concept_nodes": len(trainer.concept_node_data),
                "validated_or_promoted_concepts": sum(
                    data["status"] in {"validated", "promoted"}
                    for data in trainer.concept_node_data.values()
                ),
                "user_preferences": len(trainer.preference_node_data),
                "validated_or_promoted_fact_nodes": sum(
                    data["status"] in {"validated", "promoted"}
                    for data in trainer.fact_node_data.values()
                ),
                "abstention_patterns": len(trainer.abstention_node_data),
                "procedure_nodes": len(trainer.procedure_node_data),
                "source_documents": len(trainer.sources),
            },
            "configuration": {
                "encoder": encoder,
                "ollama_embedding_model": embedding_model if encoder == "ollama_embedding" else None,
                "phase_count": len(phases),
                "phase_names": [phase["name"] for phase in phases],
                "retention_episode_count": len(retention_episodes),
                "phase_test_episode_count": len(phase_test_episodes),
                "final_restart_exam_episode_count": len(final_exam),
                "max_accuracy_drop": max_accuracy_drop,
                "correction_matching": {
                    "query_similarity_threshold": trainer.correction_query_similarity_threshold,
                    "answer_similarity_threshold": trainer.correction_answer_similarity_threshold,
                    "combined_transfer_score_threshold": trainer.correction_transfer_score_threshold,
                },
                "fact_retrieval_similarity_threshold": trainer.fact_retrieval_similarity_threshold,
                "concept_retrieval_similarity_threshold": trainer.concept_retrieval_similarity_threshold,
                "learned_concept_bonus_by_status": {
                    "candidate": 0.0,
                    "validated": 0.05,
                    "promoted": 0.08,
                },
                "ollama_baseline_model": ollama_model,
                "ollama_baseline_mode": baseline_mode if ollama_model else None,
                "baseline_history_mode": baseline_history_mode,
                "max_prior_interactions": max_prior_interactions if baseline_history_mode == "relevant" else None,
                "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
                "baseline_history_interaction_count": len(history_episodes),
                "pretraining_corpus": {
                    "files": [str(path) for path in (pretraining_corpus_paths or ())],
                    "source_documents": len(pretraining_sources),
                    "characters": sum(len(source.text) for source in pretraining_sources.values()),
                    "source_ids": sorted(pretraining_sources),
                    "source_type_counts": {
                        source_type: sum(
                            source.metadata.get("source_type") == source_type
                            for source in pretraining_sources.values()
                        )
                        for source_type in sorted({
                            source.metadata.get("source_type", "unspecified")
                            for source in pretraining_sources.values()
                        })
                    },
                    "license_counts": {
                        license_name: sum(
                            source.metadata.get("license", "unspecified") == license_name
                            for source in pretraining_sources.values()
                        )
                        for license_name in sorted({
                            source.metadata.get("license", "unspecified")
                            for source in pretraining_sources.values()
                        })
                    },
                    "metadata_coverage": (
                        sum(bool(source.metadata) for source in pretraining_sources.values())
                        / len(pretraining_sources) if pretraining_sources else 0.0
                    ),
                    "ingested_before_first_phase_exam": bool(pretraining_sources),
                },
                "vocabulary_expansion": vocabulary_expansion,
            },
            "host": {
                "platform": platform.platform(),
                "architecture": platform.machine(),
                "processor": _processor_name(),
                "logical_cpu_count": os.cpu_count(),
                "gpus": _gpu_metadata(),
            },
            "checkpoint": str(checkpoint),
        }
        if ollama_model:
            history = _supervised_demonstrations([*history_episodes, *phase_episodes])
            if baseline_mode in {"both", "chat"}:
                result["ollama_baseline"] = _evaluate_ollama(
                    OllamaChatBaseline(ollama_model), source_list, final_exam,
                    prior_interactions=history,
                    history_mode=baseline_history_mode,
                    max_prior_interactions=max_prior_interactions,
                    progress_path=out / "ollama_baseline_progress.jsonl",
                )
            if baseline_mode in {"both", "retrieval"}:
                result["ollama_retrieval_baseline"] = _evaluate_ollama(
                    OllamaRetrievalBaseline(ollama_model), source_list, final_exam,
                    prior_interactions=history,
                    history_mode=baseline_history_mode,
                    max_prior_interactions=max_prior_interactions,
                    progress_path=out / "ollama_retrieval_progress.jsonl",
                )
        report_path = out / "cumulative_study_report.json"
        result["report"] = str(report_path)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def _merge_sources(target: dict[str, SourceDocument], sources: list[SourceDocument]) -> None:
    for source in sources:
        previous = target.get(source.source_id)
        if previous is not None and previous != source:
            raise ValueError(f"source_id reused with different content: {source.source_id}")
        target[source.source_id] = source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", action="append", type=Path, required=True)
    parser.add_argument("--retention", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=("tfidf", "ollama_embedding"), default="tfidf")
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--ollama-model")
    parser.add_argument(
        "--baseline-mode", choices=("both", "chat", "retrieval"), default="both",
        help="Ollama comparison to run; retrieval limits each prompt to the BM25 top 3 sources",
    )
    parser.add_argument("--max-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--fact-retrieval-similarity-threshold", type=float, default=0.30)
    parser.add_argument("--concept-retrieval-similarity-threshold", type=float, default=0.30)
    parser.add_argument("--correction-query-similarity-threshold", type=float)
    parser.add_argument("--correction-answer-similarity-threshold", type=float)
    parser.add_argument("--correction-transfer-score-threshold", type=float)
    parser.add_argument(
        "--no-vocabulary-expansion", action="store_false", dest="expand_vocabulary",
        help="keep the checkpoint TF-IDF vocabulary fixed during continuation",
    )
    parser.set_defaults(expand_vocabulary=True)
    parser.add_argument("--baseline-history-mode", choices=("all", "relevant"), default="relevant")
    parser.add_argument("--max-prior-interactions", type=int, default=8)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument(
        "--baseline-history-curriculum", action="append", type=Path, default=[]
    )
    parser.add_argument(
        "--pretraining-corpus", action="append", type=Path, default=[],
        help="source-only JSONL corpus ingested into HFN before supervised phases",
    )
    args = parser.parse_args()
    run_cumulative(
        args.phase,
        args.retention,
        args.output_dir,
        encoder=args.encoder,
        embedding_model=args.embedding_model,
        ollama_model=args.ollama_model,
        max_accuracy_drop=args.max_accuracy_drop,
        fact_retrieval_similarity_threshold=args.fact_retrieval_similarity_threshold,
        concept_retrieval_similarity_threshold=args.concept_retrieval_similarity_threshold,
        expand_vocabulary=args.expand_vocabulary,
        correction_query_similarity_threshold=args.correction_query_similarity_threshold,
        correction_answer_similarity_threshold=args.correction_answer_similarity_threshold,
        correction_transfer_score_threshold=args.correction_transfer_score_threshold,
        baseline_history_mode=args.baseline_history_mode,
        max_prior_interactions=args.max_prior_interactions,
        baseline_mode=args.baseline_mode,
        initial_checkpoint=args.initial_checkpoint,
        baseline_history_paths=args.baseline_history_curriculum,
        pretraining_corpus_paths=args.pretraining_corpus,
    )


if __name__ == "__main__":
    main()
