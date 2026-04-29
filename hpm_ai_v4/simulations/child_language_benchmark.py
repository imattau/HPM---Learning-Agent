"""Child-level language benchmark for early vocabulary and sentence acquisition."""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence

from hpm_ai_v4.io.adapters import SentenceAdapter
from hpm_ai_v4.simulations.build_child_language_corpus import build_child_language_corpus
from hpm_ai_v4.simulations.chat_simulation import BasicChatSession
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary
from hpm_ai_v4.tools.text_signals import TextSignalExtractor


@dataclass(frozen=True)
class ChildExample:
    prompt_text: str
    target_text: str
    category: str
    held_out: bool


DEFAULT_EXAMPLES: List[ChildExample] = [
    ChildExample("What is this?", "This is a ball.", "label", False),
    ChildExample("Where is the cup?", "The cup is here.", "location", False),
    ChildExample("Give me the toy.", "Here is the toy.", "instruction", True),
    ChildExample("Can you go?", "Yes, I can go.", "instruction", True),
    ChildExample("What does the dog do?", "The dog runs.", "action", False),
    ChildExample("Can you help me?", "Yes, I can help you.", "request", False),
]


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _warm_agent(agent: LayeredAgent, text: str) -> None:
    if getattr(agent, "surface_mode", "ascii") == "word":
        agent.observe_text(text, feedback_mode="target")
        return
    for ch in text:
        raw = 94 if ch == "\n" else ord(ch) - 32
        if 0 <= raw <= 94:
            agent.perceive(raw)


def _sentence_features(text: str) -> Dict[str, Any]:
    adapter = SentenceAdapter()
    spans = adapter.segment(text)
    sentence_count = len(spans)
    dominant = spans[0].sentence_type if spans else "fragment"
    confidence = float(sum(span.confidence for span in spans) / max(1, sentence_count)) if spans else 0.0
    return {
        "sentence_count": sentence_count,
        "dominant_sentence_type": dominant,
        "sentence_confidence": confidence,
    }


def _score_output(text: str, target_text: str) -> Dict[str, float]:
    tokens = [tok.lower() for tok in text.split() if tok.strip()]
    target_tokens = [tok.lower() for tok in target_text.split() if tok.strip()]
    if not tokens or not target_tokens:
        agreement = 0.0
    else:
        overlap = len(set(tokens) & set(target_tokens))
        agreement = overlap / float(max(len(set(target_tokens)), 1))
    sentence_features = _sentence_features(text)
    target_sentence_features = _sentence_features(target_text)
    exact = 1.0 if text.strip().lower() == target_text.strip().lower() else 0.0
    return {
        "token_agreement": agreement,
        "exact_match": exact,
        "sentence_count_match": 1.0 if sentence_features["sentence_count"] == target_sentence_features["sentence_count"] else 0.0,
        "sentence_type_match": 1.0 if sentence_features["dominant_sentence_type"] == target_sentence_features["dominant_sentence_type"] else 0.0,
        "sentence_confidence": sentence_features["sentence_confidence"],
    }


def _run_arm(
    *,
    examples: Sequence[ChildExample],
    corpus_text: str,
    use_sentence_features: bool,
    num_workers: int,
    use_dict: bool,
    warmup_chars: int,
    response_steps: int,
    history_window: int,
) -> Dict[str, Any]:
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    agent = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar, surface_mode="word")
    warmup_text = corpus_text[:warmup_chars]
    if warmup_text:
        _warm_agent(agent, warmup_text)

    session = BasicChatSession(
        agent,
        history_window=history_window,
        response_steps=response_steps,
        use_constraints=use_dict,
        text_signals=TextSignalExtractor(),
        use_sentence_features=use_sentence_features,
    )

    turns: List[Dict[str, Any]] = []
    for example in examples:
        result = session.chat_turn(example.prompt_text, target_reply=example.target_text)
        output = _score_output(result.response_text, example.target_text)
        turns.append(
            {
                "prompt_text": example.prompt_text,
                "target_text": example.target_text,
                "response_text": result.response_text,
                "category": example.category,
                "held_out": example.held_out,
                "response_stats": result.response_stats,
                "output": output,
            }
        )

    metrics = {
        "avg_token_agreement": float(mean(turn["output"]["token_agreement"] for turn in turns)) if turns else 0.0,
        "avg_exact_match": float(mean(turn["output"]["exact_match"] for turn in turns)) if turns else 0.0,
        "avg_sentence_count_match": float(mean(turn["output"]["sentence_count_match"] for turn in turns)) if turns else 0.0,
        "avg_sentence_type_match": float(mean(turn["output"]["sentence_type_match"] for turn in turns)) if turns else 0.0,
        "avg_sentence_confidence": float(mean(turn["output"]["sentence_confidence"] for turn in turns)) if turns else 0.0,
        "avg_repeat_score": float(mean(turn["response_stats"].get("repeat_score", 0.0) for turn in turns)) if turns else 0.0,
        "avg_structure_score": float(mean(turn["response_stats"].get("structure_score", 0.0) for turn in turns)) if turns else 0.0,
        "avg_commonness_score": float(mean(turn["response_stats"].get("commonness_score", 0.0) for turn in turns)) if turns else 0.0,
        "avg_text_signal_score": float(mean(turn["response_stats"].get("text_signal_score", 0.0) for turn in turns)) if turns else 0.0,
    }
    metrics["avg_sentence_shape_score"] = 0.5 * metrics["avg_sentence_count_match"] + 0.5 * metrics["avg_sentence_type_match"]
    metrics["held_out_exact_match"] = float(mean(turn["output"]["exact_match"] for turn in turns if turn["held_out"])) if any(turn["held_out"] for turn in turns) else 0.0
    metrics["held_out_token_agreement"] = float(mean(turn["output"]["token_agreement"] for turn in turns if turn["held_out"])) if any(turn["held_out"] for turn in turns) else 0.0
    return {
        "use_sentence_features": use_sentence_features,
        "turns": turns,
        "metrics": metrics,
    }


def run_child_language_benchmark(
    *,
    corpus_path: str,
    examples: Optional[Sequence[ChildExample]] = None,
    repeats: int = 1,
    seed: Optional[int] = None,
    warmup_chars: int = 200,
    response_steps: int = 24,
    history_window: int = 3,
    num_workers: int = 1,
    use_dict: bool = False,
    checkpoint_dir: str = ".",
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare child-level language behavior with and without sentence-aware scoring."""
    corpus_text = _read_text(corpus_path)
    examples = list(examples or DEFAULT_EXAMPLES)

    os.makedirs(checkpoint_dir, exist_ok=True)
    library_path = os.path.join(checkpoint_dir, "child_language_corpus.txt")
    if not os.path.exists(library_path):
        result = build_child_language_corpus(library_path)
        corpus_text = _read_text(result.corpus_path)
    repeats = max(1, int(repeats))
    rng = random.Random(seed)
    runs = {"sentence_aware": [], "sentence_blind": []}
    base_examples = list(examples)
    for run_idx in range(repeats):
        shuffled = list(base_examples)
        rng.shuffle(shuffled)
        runs["sentence_aware"].append(
            _run_arm(
                examples=shuffled,
                corpus_text=corpus_text,
                use_sentence_features=True,
                num_workers=num_workers,
                use_dict=use_dict,
                warmup_chars=warmup_chars,
                response_steps=response_steps,
                history_window=history_window,
            )
        )
        runs["sentence_blind"].append(
            _run_arm(
                examples=shuffled,
                corpus_text=corpus_text,
                use_sentence_features=False,
                num_workers=num_workers,
                use_dict=use_dict,
                warmup_chars=warmup_chars,
                response_steps=response_steps,
                history_window=history_window,
            )
        )

    def _aggregate(arm_runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        metric_keys = sorted(set().union(*(run["metrics"].keys() for run in arm_runs))) if arm_runs else []
        metrics: Dict[str, Any] = {}
        for key in metric_keys:
            values = [float(run["metrics"].get(key, 0.0)) for run in arm_runs]
            metrics[key] = {
                "mean": float(mean(values)) if values else 0.0,
                "std": float(pstdev(values)) if len(values) > 1 else 0.0,
                "min": float(min(values)) if values else 0.0,
                "max": float(max(values)) if values else 0.0,
            }
        return {
            "metrics": metrics,
            "runs": arm_runs,
        }

    arms = {
        "sentence_aware": _aggregate(runs["sentence_aware"]),
        "sentence_blind": _aggregate(runs["sentence_blind"]),
    }

    comparison = {
        "sentence_aware": arms["sentence_aware"],
        "sentence_blind": arms["sentence_blind"],
        "delta": {
            key: float(arms["sentence_aware"]["metrics"].get(key, {}).get("mean", 0.0) - arms["sentence_blind"]["metrics"].get(key, {}).get("mean", 0.0))
            for key in sorted(set(arms["sentence_aware"]["metrics"]) | set(arms["sentence_blind"]["metrics"]))
        },
        "examples": [
            {
                "prompt_text": ex.prompt_text,
                "target_text": ex.target_text,
                "category": ex.category,
                "held_out": ex.held_out,
            }
            for ex in examples
        ],
    }

    out_path = report_path or os.path.join(checkpoint_dir, "child_language_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, sort_keys=True)
    comparison["report_path"] = out_path
    return comparison


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Child-level language benchmark")
    p.add_argument("--corpus", required=True, help="Path to a child-directed corpus")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--warmup-chars", type=int, default=200)
    p.add_argument("--response-steps", type=int, default=24)
    p.add_argument("--history-window", type=int, default=3)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dict", action="store_true", help="Enable dictionary and grammar validators")
    p.add_argument("--checkpoint-dir", default=".")
    p.add_argument("--report-path", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_child_language_benchmark(
        corpus_path=args.corpus,
        repeats=args.repeats,
        seed=args.seed,
        warmup_chars=args.warmup_chars,
        response_steps=args.response_steps,
        history_window=args.history_window,
        num_workers=args.workers,
        use_dict=args.dict,
        checkpoint_dir=args.checkpoint_dir,
        report_path=args.report_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
