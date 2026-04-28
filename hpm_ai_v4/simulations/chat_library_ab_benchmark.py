"""A/B benchmark for chat library seeds."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from hpm_ai_v4.simulations.chat_simulation import BasicChatSession, _load_chat_library
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary


@dataclass(frozen=True)
class ChatLibraryRun:
    library_path: str
    responses: List[str]
    response_stats: List[Dict[str, Any]]
    prompts: List[str]
    metrics: Dict[str, float]


def _warm_agent(agent: LayeredAgent, text: str) -> None:
    for ch in text:
        raw = 94 if ch == "\n" else ord(ch) - 32
        if 0 <= raw <= 94:
            agent.perceive(raw)


def _response_diversity(responses: Sequence[str]) -> float:
    if not responses:
        return 0.0
    return len(set(responses)) / float(len(responses))


def _adjacent_repeat_rate(responses: Sequence[str]) -> float:
    if len(responses) < 2:
        return 0.0
    repeats = sum(1 for a, b in zip(responses, responses[1:]) if a == b)
    return repeats / float(len(responses) - 1)


def _mean_metric(stats: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [float(item.get(key, 0.0)) for item in stats if key in item]
    return float(mean(vals)) if vals else 0.0


def _run_single_library(
    *,
    corpus_path: str,
    prompts: Sequence[str],
    library_path: str,
    warmup_chars: int,
    response_steps: int,
    history_window: int,
    num_workers: int,
    use_dict: bool,
    seed_corpus_path: Optional[str],
) -> ChatLibraryRun:
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    agent = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)

    loaded = _load_chat_library(agent, library_path)
    if loaded:
        print(f"Loaded library from {library_path} ({loaded} bundle parts)")

    seed_source = seed_corpus_path if seed_corpus_path and os.path.exists(seed_corpus_path) else corpus_path
    with open(seed_source, "r", encoding="utf-8", errors="ignore") as f:
        warmup_text = f.read(max(warmup_chars, 1))
    if warmup_text:
        _warm_agent(agent, warmup_text)

    session = BasicChatSession(
        agent,
        history_window=history_window,
        response_steps=response_steps,
        use_constraints=use_dict,
    )

    responses: List[str] = []
    response_stats: List[Dict[str, Any]] = []
    for prompt in prompts:
        result = session.chat_turn(prompt)
        responses.append(result.response_text)
        response_stats.append(result.response_stats)

    metrics = {
        "response_diversity": _response_diversity(responses),
        "adjacent_repeat_rate": _adjacent_repeat_rate(responses),
        "avg_length": float(mean(len(text) for text in responses)) if responses else 0.0,
        "avg_plausibility": _mean_metric(response_stats, "plausibility"),
        "avg_repeat_score": _mean_metric(response_stats, "repeat_score"),
        "avg_structure_score": _mean_metric(response_stats, "structure_score"),
        "avg_commonness_score": _mean_metric(response_stats, "commonness_score"),
        "avg_text_signal_score": _mean_metric(response_stats, "text_signal_score"),
    }

    return ChatLibraryRun(
        library_path=library_path,
        responses=responses,
        response_stats=response_stats,
        prompts=list(prompts),
        metrics=metrics,
    )


def run_chat_library_ab_benchmark(
    *,
    corpus_path: str,
    prompts: Sequence[str],
    small_library_path: str,
    large_library_path: str,
    warmup_chars: int = 20,
    response_steps: int = 12,
    history_window: int = 3,
    num_workers: int = 1,
    use_dict: bool = True,
    checkpoint_dir: str = ".",
    seed_corpus_path: Optional[str] = None,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare a small and large chat seed on the same prompt list."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    runs = {
        "small": _run_single_library(
            corpus_path=corpus_path,
            prompts=prompts,
            library_path=small_library_path,
            warmup_chars=warmup_chars,
            response_steps=response_steps,
            history_window=history_window,
            num_workers=num_workers,
            use_dict=use_dict,
            seed_corpus_path=seed_corpus_path,
        ),
        "large": _run_single_library(
            corpus_path=corpus_path,
            prompts=prompts,
            library_path=large_library_path,
            warmup_chars=warmup_chars,
            response_steps=response_steps,
            history_window=history_window,
            num_workers=num_workers,
            use_dict=use_dict,
            seed_corpus_path=seed_corpus_path,
        ),
    }

    comparison = {
        "small": {
            "library_path": runs["small"].library_path,
            "responses": runs["small"].responses,
            "metrics": runs["small"].metrics,
        },
        "large": {
            "library_path": runs["large"].library_path,
            "responses": runs["large"].responses,
            "metrics": runs["large"].metrics,
        },
        "delta": {
            key: float(runs["large"].metrics.get(key, 0.0) - runs["small"].metrics.get(key, 0.0))
            for key in sorted(set(runs["small"].metrics) | set(runs["large"].metrics))
        },
        "prompts": list(prompts),
    }

    out_path = report_path or os.path.join(checkpoint_dir, "chat_library_ab_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, sort_keys=True)
    comparison["report_path"] = out_path
    return comparison


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two chat seed libraries")
    p.add_argument("--corpus", required=True, help="Path to a warmup text corpus")
    p.add_argument("--small-library", required=True, help="Base path for the small library")
    p.add_argument("--large-library", required=True, help="Base path for the large library")
    p.add_argument("--prompt", action="append", dest="prompts", help="Prompt to test; can be repeated")
    p.add_argument("--warmup-chars", type=int, default=20)
    p.add_argument("--response-steps", type=int, default=12)
    p.add_argument("--history-window", type=int, default=3)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-dict", action="store_true")
    p.add_argument("--checkpoint-dir", default=".")
    p.add_argument("--seed-corpus", default=None)
    p.add_argument("--report-path", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    prompts = args.prompts or [
        "Hello.",
        "What can you do?",
        "Why is conversation hard?",
        "Tell me something useful.",
    ]
    result = run_chat_library_ab_benchmark(
        corpus_path=args.corpus,
        prompts=prompts,
        small_library_path=args.small_library,
        large_library_path=args.large_library,
        warmup_chars=args.warmup_chars,
        response_steps=args.response_steps,
        history_window=args.history_window,
        num_workers=args.workers,
        use_dict=not args.no_dict,
        checkpoint_dir=args.checkpoint_dir,
        seed_corpus_path=args.seed_corpus,
        report_path=args.report_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
