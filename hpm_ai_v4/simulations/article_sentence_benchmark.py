"""Article-length benchmark comparing sentence-aware vs sentence-blind text scoring."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hpm_ai_v4.io.adapters import SentenceAdapter
from hpm_ai_v4.simulations.chat_simulation import BasicChatSession
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary
from hpm_ai_v4.tools.text_signals import TextSignalExtractor


@dataclass(frozen=True)
class ArticleExample:
    prompt_text: str
    target_text: str
    prompt_sentences: int
    target_sentences: int
    prompt_paragraphs: int
    target_paragraphs: int


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _sentence_windows(
    text: str,
    *,
    prompt_sentences: int = 3,
    target_sentences: int = 2,
    max_examples: int = 8,
) -> List[ArticleExample]:
    adapter = SentenceAdapter()
    paragraphs = adapter.segment_paragraphs(text)
    sentences = adapter.segment(text)
    if len(sentences) < prompt_sentences + target_sentences:
        return []

    examples: List[ArticleExample] = []
    step = max(1, target_sentences)
    for start in range(0, len(sentences) - (prompt_sentences + target_sentences) + 1, step):
        prompt_slice = sentences[start:start + prompt_sentences]
        target_slice = sentences[start + prompt_sentences:start + prompt_sentences + target_sentences]
        prompt_text = " ".join(span.text for span in prompt_slice).strip()
        target_text = " ".join(span.text for span in target_slice).strip()
        if not prompt_text or not target_text:
            continue
        prompt_paragraphs = _paragraph_count(paragraphs, prompt_text)
        target_paragraphs = _paragraph_count(paragraphs, target_text)
        examples.append(
            ArticleExample(
                prompt_text=prompt_text,
                target_text=target_text,
                prompt_sentences=len(prompt_slice),
                target_sentences=len(target_slice),
                prompt_paragraphs=prompt_paragraphs,
                target_paragraphs=target_paragraphs,
            )
        )
        if len(examples) >= max_examples:
            break
    return examples


def _paragraph_count(paragraphs, text: str) -> int:
    if not text:
        return 0
    lowered = text.strip().lower()
    count = 0
    for paragraph in paragraphs:
        if lowered and lowered in paragraph.text.lower():
            count += 1
    return count


def _warm_agent(agent: LayeredAgent, text: str) -> None:
    for ch in text:
        raw = 94 if ch == "\n" else ord(ch) - 32
        if 0 <= raw <= 94:
            agent.perceive(raw)


def _boundary_similarity(generated: str, target: str) -> Dict[str, float]:
    adapter = SentenceAdapter()
    gen_spans = adapter.segment(generated)
    target_spans = adapter.segment(target)
    gen_count = len(gen_spans)
    target_count = len(target_spans)
    count_match = 1.0 if gen_count == target_count else 1.0 - min(1.0, abs(gen_count - target_count) / float(max(gen_count, target_count, 1)))
    gen_type = gen_spans[0].sentence_type if gen_spans else "fragment"
    target_type = target_spans[0].sentence_type if target_spans else "fragment"
    type_match = 1.0 if gen_type == target_type else 0.0
    confidence = float(sum(span.confidence for span in gen_spans) / max(1, gen_count)) if gen_spans else 0.0
    paragraph_count = len(adapter.segment_paragraphs(generated))
    return {
        "generated_sentence_count": float(gen_count),
        "target_sentence_count": float(target_count),
        "sentence_count_match": float(count_match),
        "dominant_sentence_type_match": float(type_match),
        "generated_sentence_confidence": confidence,
        "generated_paragraph_count": float(paragraph_count),
    }


def _run_arm(
    *,
    examples: Sequence[ArticleExample],
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
    agent = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)
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
        boundary = _boundary_similarity(result.response_text, example.target_text)
        record = {
            "prompt_text": example.prompt_text,
            "target_text": example.target_text,
            "response_text": result.response_text,
            "response_stats": result.response_stats,
            "boundary": boundary,
        }
        turns.append(record)

    metrics = {
        "avg_target_agreement": float(mean(turn["response_stats"].get("token_agreement", 0.0) for turn in turns)) if turns else 0.0,
        "avg_plausibility": float(mean(turn["response_stats"].get("plausibility", 0.0) for turn in turns)) if turns else 0.0,
        "avg_repeat_score": float(mean(turn["response_stats"].get("repeat_score", 0.0) for turn in turns)) if turns else 0.0,
        "avg_structure_score": float(mean(turn["response_stats"].get("structure_score", 0.0) for turn in turns)) if turns else 0.0,
        "avg_commonness_score": float(mean(turn["response_stats"].get("commonness_score", 0.0) for turn in turns)) if turns else 0.0,
        "avg_text_signal_score": float(mean(turn["response_stats"].get("text_signal_score", 0.0) for turn in turns)) if turns else 0.0,
        "avg_sentence_count_match": float(mean(turn["boundary"]["sentence_count_match"] for turn in turns)) if turns else 0.0,
        "avg_dominant_sentence_type_match": float(mean(turn["boundary"]["dominant_sentence_type_match"] for turn in turns)) if turns else 0.0,
        "avg_generated_sentence_confidence": float(mean(turn["boundary"]["generated_sentence_confidence"] for turn in turns)) if turns else 0.0,
        "avg_generated_sentence_count": float(mean(turn["boundary"]["generated_sentence_count"] for turn in turns)) if turns else 0.0,
        "avg_target_sentence_count": float(mean(turn["boundary"]["target_sentence_count"] for turn in turns)) if turns else 0.0,
        "avg_generated_paragraph_count": float(mean(turn["boundary"]["generated_paragraph_count"] for turn in turns)) if turns else 0.0,
    }
    metrics["avg_sentence_structure_score"] = 0.5 * metrics["avg_sentence_count_match"] + 0.5 * metrics["avg_dominant_sentence_type_match"]
    return {
        "use_sentence_features": use_sentence_features,
        "turns": turns,
        "metrics": metrics,
    }


def run_article_sentence_benchmark(
    *,
    corpus_path: str,
    prompt_sentences: int = 3,
    target_sentences: int = 2,
    examples: int = 6,
    warmup_chars: int = 200,
    response_steps: int = 32,
    history_window: int = 4,
    num_workers: int = 1,
    use_dict: bool = False,
    checkpoint_dir: str = ".",
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare sentence-aware and sentence-blind scoring on article continuations."""
    corpus_text = _read_text(corpus_path)
    slices = _sentence_windows(
        corpus_text,
        prompt_sentences=prompt_sentences,
        target_sentences=target_sentences,
        max_examples=examples,
    )
    if not slices:
        raise ValueError("Not enough sentence windows in corpus")

    os.makedirs(checkpoint_dir, exist_ok=True)
    arms = {
        "sentence_aware": _run_arm(
            examples=slices,
            corpus_text=corpus_text,
            use_sentence_features=True,
            num_workers=num_workers,
            use_dict=use_dict,
            warmup_chars=warmup_chars,
            response_steps=response_steps,
            history_window=history_window,
        ),
        "sentence_blind": _run_arm(
            examples=slices,
            corpus_text=corpus_text,
            use_sentence_features=False,
            num_workers=num_workers,
            use_dict=use_dict,
            warmup_chars=warmup_chars,
            response_steps=response_steps,
            history_window=history_window,
        ),
    }

    comparison = {
        "sentence_aware": arms["sentence_aware"]["metrics"],
        "sentence_blind": arms["sentence_blind"]["metrics"],
        "delta": {
            key: float(arms["sentence_aware"]["metrics"].get(key, 0.0) - arms["sentence_blind"]["metrics"].get(key, 0.0))
            for key in sorted(set(arms["sentence_aware"]["metrics"]) | set(arms["sentence_blind"]["metrics"]))
        },
        "examples": [
            {
                "prompt_text": ex.prompt_text,
                "target_text": ex.target_text,
                "prompt_sentences": ex.prompt_sentences,
                "target_sentences": ex.target_sentences,
                "prompt_paragraphs": ex.prompt_paragraphs,
                "target_paragraphs": ex.target_paragraphs,
            }
            for ex in slices
        ],
    }

    out_path = report_path or os.path.join(checkpoint_dir, "article_sentence_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, sort_keys=True)
    comparison["report_path"] = out_path
    return comparison


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Article sentence-aware benchmark")
    p.add_argument("--corpus", required=True, help="Path to a long article-like corpus")
    p.add_argument("--prompt-sentences", type=int, default=3)
    p.add_argument("--target-sentences", type=int, default=2)
    p.add_argument("--examples", type=int, default=6)
    p.add_argument("--warmup-chars", type=int, default=200)
    p.add_argument("--response-steps", type=int, default=32)
    p.add_argument("--history-window", type=int, default=4)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dict", action="store_true", help="Enable dictionary and grammar validators")
    p.add_argument("--checkpoint-dir", default=".")
    p.add_argument("--report-path", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_article_sentence_benchmark(
        corpus_path=args.corpus,
        prompt_sentences=args.prompt_sentences,
        target_sentences=args.target_sentences,
        examples=args.examples,
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
