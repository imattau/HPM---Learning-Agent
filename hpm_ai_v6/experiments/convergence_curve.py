import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.append(os.getcwd())

from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader


def split_sentences(reader: MultiAgentReader, max_chunks: int | None, max_words_per_chunk: int | None) -> List[str]:
    documents = reader._load_documents(max_chunks=max_chunks, max_words_per_chunk=max_words_per_chunk)
    return [sentence for document in documents for sentence in document]


def run_curve(
    corpus_path: str,
    mode: str,
    budgets: List[int],
    holdout_ratio: float,
    max_chunks: int | None,
    max_words_per_chunk: int | None,
) -> Dict[str, object]:
    base_reader = MultiAgentReader(corpus_path)
    sentences = split_sentences(base_reader, max_chunks=max_chunks, max_words_per_chunk=max_words_per_chunk)
    if len(sentences) < 2:
        raise ValueError("Need at least 2 sentences to build a convergence curve")

    split_idx = max(1, int(len(sentences) * (1.0 - holdout_ratio)))
    if split_idx >= len(sentences):
        split_idx = len(sentences) - 1

    train_sentences = sentences[:split_idx]
    holdout_sentences = sentences[split_idx:]

    points = []
    for budget in budgets:
        subset = train_sentences[: min(budget, len(train_sentences))]
        reader = MultiAgentReader(corpus_path)
        if mode == "active":
            reader.train_sequence_active(subset, enable_causal=False)
        else:
            reader.train_sequence(subset, enable_causal=False)
        metrics = reader.evaluate_sentences(holdout_sentences)
        points.append(
            {
                "budget": len(subset),
                "metrics": metrics,
                "char_patterns": len(reader.char_agent.patterns),
                "word_patterns": len(reader.word_agent.patterns),
                "phrase_patterns": len(reader.phrase_agent.patterns),
                "semantic_patterns": len(reader.semantic_agent.patterns),
            }
        )

    return {
        "mode": mode,
        "corpus_path": corpus_path,
        "train_sentences": len(train_sentences),
        "holdout_sentences": len(holdout_sentences),
        "points": points,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate convergence-curve metrics for V6 reader training.")
    parser.add_argument("--corpus", default="library_bootstrap/chat_ultra")
    parser.add_argument("--mode", choices=["sequential", "active", "both"], default="both")
    parser.add_argument("--budgets", default="1,2,4,8")
    parser.add_argument("--holdout-ratio", type=float, default=0.25)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--max-words-per-chunk", type=int, default=32)
    args = parser.parse_args()

    budgets = [int(part) for part in args.budgets.split(",") if part.strip()]
    modes = ["sequential", "active"] if args.mode == "both" else [args.mode]

    results = [
        run_curve(
            corpus_path=args.corpus,
            mode=mode,
            budgets=budgets,
            holdout_ratio=args.holdout_ratio,
            max_chunks=args.max_chunks,
            max_words_per_chunk=args.max_words_per_chunk,
        )
        for mode in modes
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
