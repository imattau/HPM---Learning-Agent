"""Build an HFN checkpoint from source-only corpus material before supervision."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.domains.ollama_embedding import OllamaEmbeddingTextDomainConfig
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.training.study import (
    SupervisedStudyTrainer,
    _sentences_with_offsets,
    load_curriculum_jsonl,
)


def pretrain_corpus(
    corpus_paths: Iterable[str | Path], output_dir: str | Path,
    *, encoder: str = "tfidf", embedding_model: str = "qwen3-embedding:0.6b",
    max_vocab: int = 20_000,
) -> dict:
    sources_by_id = {}
    inputs = [Path(path) for path in corpus_paths]
    if not inputs:
        raise ValueError("provide at least one source-only corpus")
    for path in inputs:
        sources, episodes = load_curriculum_jsonl(path)
        if not sources:
            raise ValueError(f"pretraining corpus has no sources: {path}")
        if episodes:
            raise ValueError(f"pretraining corpus must contain source rows only: {path}")
        for source in sources:
            previous = sources_by_id.get(source.source_id)
            if previous is not None and previous != source:
                raise ValueError(f"conflicting duplicate source ID: {source.source_id}")
            sources_by_id[source.source_id] = source
    sources = list(sources_by_id.values())
    if encoder == "tfidf":
        config = TextDomainConfig.from_passages(
            [source.text for source in sources], max_vocab=max_vocab,
        )
    elif encoder == "ollama_embedding":
        config = OllamaEmbeddingTextDomainConfig(model=embedding_model)
        embedding_texts = [source.text for source in sources]
        embedding_texts.extend(
            sentence for source in sources
            for sentence, _, _ in _sentences_with_offsets(source.text)
        )
        config.precompute(embedding_texts)
    else:
        raise ValueError("encoder must be 'tfidf' or 'ollama_embedding'")
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    agent = ReaderAgent(
        config, dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(checkpoint_dir / "forest"),
    )
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest(sources)
    trainer.save_checkpoint(checkpoint_dir)

    checkpoint_files = [
        path for path in checkpoint_dir.rglob("*") if path.is_file()
    ]
    pattern_count = len(agent.patterns)
    pattern_reference_count = sum(
        pattern.id in agent.forest for pattern in agent.patterns.values()
    )
    relation_type_counts = Counter(
        getattr(pattern, "relation_type", None) or "untyped"
        for pattern in agent.patterns.values()
    )
    shared_term_rows = [
        sentence_ids for sentence_ids in getattr(agent, "term_sentences", {}).values()
        if len(sentence_ids) >= 2
    ]
    observer_compressed_count = sum(
        pattern.id.startswith("compressed(") for pattern in agent.patterns.values()
    )

    source_types = Counter(
        source.metadata.get("source_type", "unspecified") for source in sources
    )
    licenses = Counter(
        source.metadata.get("license", "unspecified") for source in sources
    )
    report = {
        "stage": "source-only-hfn-pretraining",
        "corpus_files": [str(path) for path in inputs],
        "source_documents": len(sources),
        "characters": sum(len(source.text) for source in sources),
        "source_type_counts": dict(sorted(source_types.items())),
        "license_counts": dict(sorted(licenses.items())),
        "metadata_coverage": sum(bool(source.metadata) for source in sources) / len(sources),
        "hf_dataset_ids": sorted({
            source.metadata["huggingface_dataset_id"]
            for source in sources if source.metadata.get("huggingface_dataset_id")
        }),
        "vocabulary_size": len(config.concepts),
        "encoder": encoder,
        "embedding_model": embedding_model if encoder == "ollama_embedding" else None,
        "fact_nodes_before_supervision": len(trainer.fact_node_data),
        "fit_episode_count": len(trainer._fit_episode_ids),
        "checkpoint_bytes": sum(path.stat().st_size for path in checkpoint_files),
        "agent_state_bytes": (checkpoint_dir / "agent_state.pkl").stat().st_size,
        "forest_node_count": len(agent.forest),
        "pattern_count": pattern_count,
        "compact_pattern_reference_count": pattern_reference_count,
        "pattern_relation_type_counts": dict(sorted(relation_type_counts.items())),
        "shared_term_node_count": relation_type_counts.get("term", 0),
        "shared_term_sentence_links": sum(map(len, shared_term_rows)),
        "shared_term_reuse_ratio": (
            sum(map(len, shared_term_rows)) / len(shared_term_rows)
            if shared_term_rows else 0.0
        ),
        "observer_compressed_node_count": observer_compressed_count,
        "checkpoint": str(checkpoint_dir),
    }
    (checkpoint_dir / "pretraining_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-vocab", type=int, default=20_000)
    parser.add_argument("--encoder", choices=("tfidf", "ollama_embedding"), default="tfidf")
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    args = parser.parse_args()
    print(json.dumps(pretrain_corpus(
        args.corpus, args.output_dir, max_vocab=args.max_vocab,
        encoder=args.encoder, embedding_model=args.embedding_model,
    ), indent=2))


if __name__ == "__main__":
    main()
