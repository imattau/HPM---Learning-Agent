"""Paired benchmark for word-level vs ascii-level L1 surfaces."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from hpm_ai_v4.io.adapters import SentenceAdapter
from hpm_ai_v4.simulations.child_language_benchmark import ChildExample, DEFAULT_EXAMPLES as CHILD_DEFAULT_EXAMPLES
from hpm_ai_v4.simulations.chat_simulation import BasicChatSession
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary
from hpm_ai_v4.tools.text_signals import TextSignalExtractor
from hpm_ai_v4.tools.serializer import PatternSerializer


@dataclass(frozen=True)
class WordSurfaceExample:
    prompt_text: str
    target_text: str
    prompt_sentences: int
    target_sentences: int


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _sentence_examples(
    text: str,
    *,
    prompt_sentences: int = 2,
    target_sentences: int = 1,
    max_examples: int = 8,
) -> List[WordSurfaceExample]:
    adapter = SentenceAdapter()
    sentences = adapter.segment(text)
    if len(sentences) < prompt_sentences + target_sentences:
        return []

    examples: List[WordSurfaceExample] = []
    step = max(1, target_sentences)
    for start in range(0, len(sentences) - (prompt_sentences + target_sentences) + 1, step):
        prompt_slice = sentences[start:start + prompt_sentences]
        target_slice = sentences[start + prompt_sentences:start + prompt_sentences + target_sentences]
        prompt_text = " ".join(span.text for span in prompt_slice).strip()
        target_text = " ".join(span.text for span in target_slice).strip()
        if not prompt_text or not target_text:
            continue
        examples.append(
            WordSurfaceExample(
                prompt_text=prompt_text,
                target_text=target_text,
                prompt_sentences=len(prompt_slice),
                target_sentences=len(target_slice),
            )
        )
        if len(examples) >= max_examples:
            break
    return examples


def _warm_agent(agent: LayeredAgent, text: str) -> None:
    agent.observe_text(text, feedback_mode="target")


def _load_seed_library(agent: LayeredAgent, library_path: Optional[str]) -> int:
    if not library_path or not os.path.exists(library_path):
        return 0
    if os.path.exists(library_path + ".l1.pkl"):
        return agent.load_bundle(library_path)
    surface_path = library_path + ".surface.json"
    if os.path.exists(surface_path):
        return agent.load_bundle(library_path)
    patterns = PatternSerializer.load(library_path)
    if patterns:
        agent.l1.patterns = patterns
        if patterns:
            obs_dim = getattr(patterns[0], "obs_dim", agent._adapter.obs_dim)
            inferred_surface = "word" if obs_dim > 95 else "ascii" if obs_dim == 95 else "coarse"
            agent._set_surface_mode(inferred_surface)
        return 1
    return 0


def _mean_metric(stats: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [float(item.get(key, 0.0)) for item in stats if key in item]
    return float(mean(vals)) if vals else 0.0


def _response_diversity(responses: Sequence[str]) -> float:
    if not responses:
        return 0.0
    return len(set(responses)) / float(len(responses))


def _adjacent_repeat_rate(responses: Sequence[str]) -> float:
    if len(responses) < 2:
        return 0.0
    repeats = sum(1 for a, b in zip(responses, responses[1:]) if a == b)
    return repeats / float(len(responses) - 1)


def _continuation_metrics(turns: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    metrics = {
        "avg_token_agreement": float(mean(turn["stats"].get("token_agreement", 0.0) for turn in turns)) if turns else 0.0,
        "avg_plausibility": float(mean(turn["stats"].get("plausibility", 0.0) for turn in turns)) if turns else 0.0,
        "avg_sentence_structure_score": float(mean(turn["stats"].get("sentence_structure_score", 0.0) for turn in turns)) if turns else 0.0,
        "avg_sentence_count_match": float(mean(turn["stats"].get("sentence_count_match", 0.0) for turn in turns)) if turns else 0.0,
        "avg_dominant_sentence_type_match": float(mean(turn["stats"].get("dominant_sentence_type_match", 0.0) for turn in turns)) if turns else 0.0,
    }
    metrics["avg_text_signal_score"] = float(mean(turn["stats"].get("text_signal_score", 0.0) for turn in turns)) if turns else 0.0
    return metrics


def _chat_metrics(turns: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    responses = [turn["response_text"] for turn in turns]
    metrics = {
        "response_diversity": _response_diversity(responses),
        "adjacent_repeat_rate": _adjacent_repeat_rate(responses),
        "avg_response_length": float(mean(len(text) for text in responses)) if responses else 0.0,
        "avg_plausibility": _mean_metric([turn["stats"] for turn in turns], "plausibility"),
        "avg_repeat_score": _mean_metric([turn["stats"] for turn in turns], "repeat_score"),
        "avg_structure_score": _mean_metric([turn["stats"] for turn in turns], "structure_score"),
        "avg_commonness_score": _mean_metric([turn["stats"] for turn in turns], "commonness_score"),
        "avg_text_signal_score": _mean_metric([turn["stats"] for turn in turns], "text_signal_score"),
    }
    return metrics


def _run_arm(
    *,
    corpus_text: str,
    examples: Sequence[WordSurfaceExample],
    surface_mode: str,
    num_workers: int,
    use_dict: bool,
    warmup_chars: int,
    response_steps: int,
    history_window: int,
    chat_prompts: Sequence[str],
    seed_library_path: Optional[str] = None,
) -> Dict[str, Any]:
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    agent = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar, surface_mode=surface_mode)
    loaded = _load_seed_library(agent, seed_library_path)
    if loaded:
        print(f"Loaded seed library from {seed_library_path} ({loaded} bundle parts)")
    if warmup_chars > 0:
        _warm_agent(agent, corpus_text[:warmup_chars])

    continuation_turns: List[Dict[str, Any]] = []
    for example in examples:
        planned = agent.plan_text_continuation(
            target_text=example.target_text,
            seed_text=example.prompt_text,
            horizon=response_steps,
            strategy="beam",
        )
        stats = agent.evaluate_generated_text(planned, example.target_text)
        sentence_stats = SentenceAdapter().segment(planned)
        stats.update({
            "sentence_count": float(len(sentence_stats)),
            "sentence_confidence": float(mean(span.confidence for span in sentence_stats)) if sentence_stats else 0.0,
        })
        continuation_turns.append({
            "prompt_text": example.prompt_text,
            "target_text": example.target_text,
            "planned_text": planned,
            "stats": stats,
        })

    session = BasicChatSession(
        agent,
        history_window=history_window,
        response_steps=response_steps,
        use_constraints=use_dict,
        text_signals=TextSignalExtractor(),
        use_sentence_features=True,
    )
    chat_turns: List[Dict[str, Any]] = []
    for prompt in chat_prompts:
        result = session.chat_turn(prompt)
        chat_turns.append({
            "prompt_text": prompt,
            "response_text": result.response_text,
            "stats": result.response_stats,
        })

    return {
        "surface_mode": surface_mode,
        "continuation": {
            "turns": continuation_turns,
            "metrics": _continuation_metrics(continuation_turns),
        },
        "chat": {
            "turns": chat_turns,
            "metrics": _chat_metrics(chat_turns),
        },
    }


def _score_child_output(text: str, target_text: str) -> Dict[str, float]:
    tokens = [tok.lower() for tok in text.split() if tok.strip()]
    target_tokens = [tok.lower() for tok in target_text.split() if tok.strip()]
    if not tokens or not target_tokens:
        agreement = 0.0
    else:
        overlap = len(set(tokens) & set(target_tokens))
        agreement = overlap / float(max(len(set(target_tokens)), 1))
    sentence_adapter = SentenceAdapter()
    sentence_features = sentence_adapter.segment(text)
    target_sentence_features = sentence_adapter.segment(target_text)
    return {
        "token_agreement": agreement,
        "exact_match": 1.0 if text.strip().lower() == target_text.strip().lower() else 0.0,
        "sentence_count_match": 1.0 if len(sentence_features) == len(target_sentence_features) else 0.0,
        "sentence_type_match": 1.0 if (sentence_features[0].sentence_type if sentence_features else "fragment") == (target_sentence_features[0].sentence_type if target_sentence_features else "fragment") else 0.0,
    }


def _run_child_arm(
    *,
    corpus_text: str,
    examples: Sequence[ChildExample],
    surface_mode: str,
    num_workers: int,
    use_dict: bool,
    warmup_chars: int,
    response_steps: int,
    history_window: int,
    seed_library_path: Optional[str] = None,
) -> Dict[str, Any]:
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    agent = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar, surface_mode=surface_mode)
    loaded = _load_seed_library(agent, seed_library_path)
    if loaded:
        print(f"Loaded seed library from {seed_library_path} ({loaded} bundle parts)")
    if warmup_chars > 0:
        _warm_agent(agent, corpus_text[:warmup_chars])

    session = BasicChatSession(
        agent,
        history_window=history_window,
        response_steps=response_steps,
        use_constraints=use_dict,
        text_signals=TextSignalExtractor(),
        use_sentence_features=True,
    )

    turns: List[Dict[str, Any]] = []
    for example in examples:
        result = session.chat_turn(example.prompt_text, target_reply=example.target_text)
        output = _score_child_output(result.response_text, example.target_text)
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
        "avg_repeat_score": float(mean(turn["response_stats"].get("repeat_score", 0.0) for turn in turns)) if turns else 0.0,
        "avg_structure_score": float(mean(turn["response_stats"].get("structure_score", 0.0) for turn in turns)) if turns else 0.0,
        "avg_commonness_score": float(mean(turn["response_stats"].get("commonness_score", 0.0) for turn in turns)) if turns else 0.0,
        "avg_text_signal_score": float(mean(turn["response_stats"].get("text_signal_score", 0.0) for turn in turns)) if turns else 0.0,
        "held_out_exact_match": float(mean(turn["output"]["exact_match"] for turn in turns if turn["held_out"])) if any(turn["held_out"] for turn in turns) else 0.0,
        "held_out_token_agreement": float(mean(turn["output"]["token_agreement"] for turn in turns if turn["held_out"])) if any(turn["held_out"] for turn in turns) else 0.0,
    }
    metrics["avg_sentence_shape_score"] = 0.5 * metrics["avg_sentence_count_match"] + 0.5 * metrics["avg_sentence_type_match"]
    return {
        "surface_mode": surface_mode,
        "turns": turns,
        "metrics": metrics,
    }


def run_word_surface_ab_benchmark(
    *,
    corpus_path: str,
    prompts: Optional[Sequence[str]] = None,
    prompt_sentences: int = 2,
    target_sentences: int = 1,
    examples: int = 6,
    warmup_chars: int = 200,
    response_steps: int = 24,
    history_window: int = 4,
    num_workers: int = 1,
    use_dict: bool = False,
    checkpoint_dir: str = ".",
    report_path: Optional[str] = None,
    word_library_path: Optional[str] = None,
    ascii_library_path: Optional[str] = None,
    include_child: bool = True,
    include_chat: bool = True,
) -> Dict[str, Any]:
    """Compare word and ascii surface modes on the same continuation/chat prompts."""
    corpus_text = _read_text(corpus_path)
    article_examples = _sentence_examples(
        corpus_text,
        prompt_sentences=max(3, prompt_sentences + 1),
        target_sentences=max(2, target_sentences + 1),
        max_examples=examples,
    )
    if not article_examples:
        article_examples = _sentence_examples(
            corpus_text,
            prompt_sentences=prompt_sentences,
            target_sentences=target_sentences,
            max_examples=examples,
        )
    if not article_examples:
        raise ValueError("Not enough sentence windows in corpus")

    child_examples = list(CHILD_DEFAULT_EXAMPLES) if include_child else []
    chat_prompts = list(prompts or [
        "Why is the model repeating itself?",
        "What changes when the corpus gets larger?",
        "Can you explain the main bottleneck?",
        "What should improve first?",
        "Give a brief answer.",
    ]) if include_chat else []

    os.makedirs(checkpoint_dir, exist_ok=True)
    arms = {
        "word": _run_arm(
            corpus_text=corpus_text,
            examples=article_examples,
            surface_mode="word",
            num_workers=num_workers,
            use_dict=use_dict,
            warmup_chars=warmup_chars,
            response_steps=response_steps,
            history_window=history_window,
            chat_prompts=chat_prompts,
            seed_library_path=word_library_path,
        ),
        "ascii": _run_arm(
            corpus_text=corpus_text,
            examples=article_examples,
            surface_mode="ascii",
            num_workers=num_workers,
            use_dict=use_dict,
            warmup_chars=warmup_chars,
            response_steps=response_steps,
            history_window=history_window,
            chat_prompts=chat_prompts,
            seed_library_path=ascii_library_path,
        ),
    }

    child_arms = {}
    if include_child:
        child_arms = {
            "word": _run_child_arm(
                corpus_text=corpus_text,
                examples=child_examples,
                surface_mode="word",
                num_workers=num_workers,
                use_dict=use_dict,
                warmup_chars=warmup_chars,
                response_steps=response_steps,
                history_window=history_window,
                seed_library_path=word_library_path,
            ),
            "ascii": _run_child_arm(
                corpus_text=corpus_text,
                examples=child_examples,
                surface_mode="ascii",
                num_workers=num_workers,
                use_dict=use_dict,
                warmup_chars=warmup_chars,
                response_steps=response_steps,
                history_window=history_window,
                seed_library_path=ascii_library_path,
            ),
        }

    comparison = {
        "word": arms["word"],
        "ascii": arms["ascii"],
        "child": child_arms,
        "delta": {
            f"continuation_{key}": float(arms["word"]["continuation"]["metrics"].get(key, 0.0) - arms["ascii"]["continuation"]["metrics"].get(key, 0.0))
            for key in sorted(set(arms["word"]["continuation"]["metrics"]) | set(arms["ascii"]["continuation"]["metrics"]))
        } | {
            f"chat_{key}": float(arms["word"]["chat"]["metrics"].get(key, 0.0) - arms["ascii"]["chat"]["metrics"].get(key, 0.0))
            for key in sorted(set(arms["word"]["chat"]["metrics"]) | set(arms["ascii"]["chat"]["metrics"]))
        } | (
            {
                f"child_{key}": float(child_arms["word"]["metrics"].get(key, 0.0) - child_arms["ascii"]["metrics"].get(key, 0.0))
                for key in sorted(set(child_arms["word"]["metrics"]) | set(child_arms["ascii"]["metrics"]))
            } if child_arms else {}
        ),
        "examples": [
            {
                "prompt_text": ex.prompt_text,
                "target_text": ex.target_text,
                "prompt_sentences": ex.prompt_sentences,
                "target_sentences": ex.target_sentences,
            }
            for ex in article_examples
        ],
        "child_examples": [
            {
                "prompt_text": ex.prompt_text,
                "target_text": ex.target_text,
                "category": ex.category,
                "held_out": ex.held_out,
            }
            for ex in child_examples
        ],
        "chat_prompts": chat_prompts,
        "preferred_arm": "word" if (
            arms["word"]["continuation"]["metrics"].get("avg_token_agreement", 0.0)
            + (child_arms["word"]["metrics"].get("avg_token_agreement", 0.0) if child_arms else 0.0)
            + (arms["word"]["chat"]["metrics"].get("response_diversity", 0.0) if include_chat else 0.0)
            - (arms["word"]["chat"]["metrics"].get("adjacent_repeat_rate", 0.0) if include_chat else 0.0)
        ) >= (
            arms["ascii"]["continuation"]["metrics"].get("avg_token_agreement", 0.0)
            + (child_arms["ascii"]["metrics"].get("avg_token_agreement", 0.0) if child_arms else 0.0)
            + (arms["ascii"]["chat"]["metrics"].get("response_diversity", 0.0) if include_chat else 0.0)
            - (arms["ascii"]["chat"]["metrics"].get("adjacent_repeat_rate", 0.0) if include_chat else 0.0)
        ) else "ascii",
    }

    out_path = report_path or os.path.join(checkpoint_dir, "word_surface_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, sort_keys=True)
    comparison["report_path"] = out_path
    return comparison


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare word and ascii surface modes")
    p.add_argument("--corpus", required=True, help="Path to a long article or dialogue corpus")
    p.add_argument("--prompt", action="append", dest="prompts", help="Chat prompt to test; can be repeated")
    p.add_argument("--prompt-sentences", type=int, default=2)
    p.add_argument("--target-sentences", type=int, default=1)
    p.add_argument("--examples", type=int, default=6)
    p.add_argument("--warmup-chars", type=int, default=200)
    p.add_argument("--response-steps", type=int, default=24)
    p.add_argument("--history-window", type=int, default=4)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-dict", action="store_true")
    p.add_argument("--checkpoint-dir", default=".")
    p.add_argument("--report-path", default=None)
    p.add_argument("--word-library", default=None, help="Optional seed bundle for the word arm")
    p.add_argument("--ascii-library", default=None, help="Optional seed bundle for the ascii arm")
    p.add_argument("--no-child", action="store_true", help="Skip the child-language probe")
    p.add_argument("--no-chat", action="store_true", help="Skip the chat probe")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_word_surface_ab_benchmark(
        corpus_path=args.corpus,
        prompts=args.prompts,
        prompt_sentences=args.prompt_sentences,
        target_sentences=args.target_sentences,
        examples=args.examples,
        warmup_chars=args.warmup_chars,
        response_steps=args.response_steps,
        history_window=args.history_window,
        num_workers=args.workers,
        use_dict=not args.no_dict,
        checkpoint_dir=args.checkpoint_dir,
        report_path=args.report_path,
        word_library_path=args.word_library,
        ascii_library_path=args.ascii_library,
        include_child=not args.no_child,
        include_chat=not args.no_chat,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
