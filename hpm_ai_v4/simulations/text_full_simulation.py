"""Hybrid text full simulation: target-driven generation plus optional self-feedback."""
import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np

from hpm_ai_v4.simulations.full_simulation import WikipediaStream, _stack_stage
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary
from hpm_ai_v4.tools.serializer import PatternSerializer


def _ids_to_text(ids: List[int]) -> str:
    return "".join(chr(v + 32) for v in ids if 0 <= v <= 94)


def _text_metrics_snapshot(
    layered: LayeredAgent,
    recent_chars: List[int],
    step: int,
    generated_text: str,
    target_text: str,
    target_stats: Dict[str, Any],
) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "step": step,
        "generated_text": generated_text,
        "target_text": target_text,
    }

    l1 = layered.l1_metrics()
    l2 = layered.l2_metrics(list(layered._l1_state_history[-200:]))
    l3 = layered.l3_metrics(list(layered._l2_state_history[-200:]))

    snap["compression_mi"] = l1["mi"]
    snap["pop_size"] = l1["pop_size"]
    snap["best_weight"] = l1["best_weight"]
    snap["best_loss"] = float(min(p.running_loss for p in layered.l1.patterns)) if layered.l1.patterns else 0.0
    snap["l1_accuracy"] = _class_accuracy(layered, recent_chars)
    snap["accuracy"] = snap["l1_accuracy"]
    snap["l2_accuracy"] = l2["accuracy"]
    snap["l2_pop_size"] = l2["pop_size"]
    snap["l2_compression_mi"] = l2["mi"]
    snap["l3_accuracy"] = l3["accuracy"]
    snap["l3_pop_size"] = l3["pop_size"]
    snap["l3_compression_mi"] = l3["mi"]
    snap["target_agreement"] = target_stats.get("token_agreement", 0.0)
    snap["plausibility"] = target_stats.get("plausibility", 0.0)
    snap["target_chars"] = target_stats.get("target_chars", 0)
    snap["self_chars"] = target_stats.get("self_chars", 0)
    snap["dev_stage"] = _stack_stage(
        l1_mi=snap["compression_mi"],
        l2_acc=snap["l2_accuracy"],
        l2_mi=snap["l2_compression_mi"],
        l3_acc=snap["l3_accuracy"],
        l3_mi=snap["l3_compression_mi"],
    )
    return snap


def _class_accuracy(layered: LayeredAgent, recent_chars: List[int]) -> float:
    if len(recent_chars) < 2:
        return 0.0
    adapter = layered._adapter
    correct = 0
    total = max(1, len(recent_chars) - 1)
    for i in range(len(recent_chars) - 1):
        ctx_cls = [adapter.encode(v) for v in recent_chars[max(0, i - 20):i]]
        actual_cls = adapter.encode(recent_chars[i + 1])
        relevant = layered.l1.reasoner.get_relevant_patterns(ctx_cls, top_k=3)
        if relevant:
            dist = layered.l1.reasoner.compose_predictions(relevant, ctx_cls)
            if int(np.argmax(dist)) == actual_cls:
                correct += 1
    return correct / total


def _text_benchmark_report(history: List[Dict[str, Any]]) -> None:
    if not history:
        print("No metrics recorded.")
        return

    final = history[-1]
    targets = [
        ("Target agreement", "target_agreement", 0.10),
        ("Plausibility", "plausibility", 0.35),
        ("L1 Compression MI", "compression_mi", 0.10),
        ("L2 Compression MI", "l2_compression_mi", 0.10),
        ("L3 Compression MI", "l3_compression_mi", 0.10),
        ("Population survived", "pop_size", 2),
    ]

    print("\n" + "=" * 60)
    print("TEXT BENCHMARK REPORT")
    print("=" * 60)
    print(f"{'Metric':<28} {'Target':>8} {'Final':>8} {'':>6}")
    print("-" * 60)
    all_pass = True
    for label, key, target in targets:
        val = final.get(key, 0.0)
        passed = val > target
        if not passed:
            all_pass = False
        mark = "PASS" if passed else "FAIL"
        print(f"{label:<28} {target:>8.2f} {val:>8.2f} {mark:>6}")

    print("=" * 60)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()


def run_text_simulation(
    corpus_path: str,
    total_steps: int = 20_000,
    log_every: int = 1_000,
    chunk_size: int = 120,
    warmup_chars: int = 500,
    num_workers: int = 1,
    use_dict: bool = True,
    surface_mode: str = "word",
    library_path: Optional[str] = None,
    checkpoint_dir: str = ".",
) -> List[Dict[str, Any]]:
    """Run a target-driven text simulation with optional self-feedback."""
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    layered = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar, surface_mode=surface_mode)

    if library_path and os.path.exists(library_path + ".l1.pkl"):
        layered.load_bundle(library_path)
        print(f"Loaded library from {library_path}")

    stream = WikipediaStream(corpus_path)
    stream_iter = iter(stream)
    needed = warmup_chars + total_steps + chunk_size
    raw_ids = [next(stream_iter) for _ in range(needed)]
    corpus_text = _ids_to_text(raw_ids)

    warmup_text = corpus_text[:warmup_chars]
    if warmup_text:
        layered.observe_text(warmup_text, feedback_mode="target")

    history: List[Dict[str, Any]] = []
    print(
        f"Starting text simulation: steps={total_steps} chunk={chunk_size} warmup={warmup_chars} "
        f"workers={num_workers} dict={use_dict}"
    )

    cursor = warmup_chars
    while cursor < warmup_chars + total_steps:
        target_text = corpus_text[cursor:cursor + chunk_size]
        if not target_text:
            break

        context_start = max(0, cursor - 240)
        context_text = corpus_text[context_start:cursor]
        target_text = corpus_text[cursor:cursor + chunk_size]
        planned_text = layered.plan_text_continuation(
            target_text=target_text,
            seed_text=context_text[-200:] if context_text else None,
            horizon=max(8, min(80, len(target_text) // 2)),
            strategy="beam",
            lookback=240,
        )
        if use_dict:
            generated_text = layered.generate_constrained_text(
                steps=max(8, min(80, len(target_text) // 2)),
                seed_text=context_text[-200:] if context_text else None,
                target_text=target_text,
                mode="target",
                include_seed=False,
                allowed_words=set(layered.dictionary.words) if layered.dictionary else None,
                strict_dictionary=True,
                strict_grammar=True,
            )
        else:
            generated_text = layered.generate_text(
                steps=max(8, min(80, len(target_text) // 2)),
                seed_text=context_text[-200:] if context_text else None,
                target_text=target_text,
                mode="target",
                include_seed=False,
            )
        planned_stats = layered.evaluate_generated_text(planned_text, target_text) if planned_text else {"token_agreement": 0.0, "plausibility": 0.0}
        target_stats = layered.evaluate_generated_text(generated_text, target_text)
        target_stats["planned_agreement"] = planned_stats["token_agreement"]
        target_stats["planned_plausibility"] = planned_stats["plausibility"]
        target_stats.update(
            layered.observe_text(
                target_text,
                feedback_mode="hybrid",
                generated_text=generated_text,
                self_feedback_weight=0.02,
            )
        )

        snap = _text_metrics_snapshot(
            layered=layered,
            recent_chars=raw_ids[max(0, cursor - 240):cursor + len(target_text)],
            step=cursor,
            generated_text=generated_text,
            target_text=target_text,
            target_stats=target_stats,
        )
        history.append(snap)

        if cursor % log_every == 0 or cursor == warmup_chars:
            print(
                f"[step {cursor:6d}] "
                f"gen_agree={snap['target_agreement']:.3f} plaus={snap['plausibility']:.3f} "
                f"L1 mi={snap['compression_mi']:.3f} acc={snap['l1_accuracy']:.3f} "
                f"L2 acc={snap['l2_accuracy']:.3f} mi={snap['l2_compression_mi']:.3f} | "
                f"L3 acc={snap['l3_accuracy']:.3f} mi={snap['l3_compression_mi']:.3f} "
                f"stage={snap['dev_stage']}"
            )
            print(f"  target:    {target_text[:120]!r}")
            print(f"  planned:   {planned_text[:120]!r}")
            print(f"  plan_ag:   {planned_stats['token_agreement']:.3f} plan_pl: {planned_stats['plausibility']:.3f}")
            print(f"  generated: {generated_text[:120]!r}")

        cursor += chunk_size

    os.makedirs(checkpoint_dir, exist_ok=True)
    final_base = os.path.join(checkpoint_dir, "final_text_library")
    layered.save_bundle(final_base)
    print(f"Final text library saved: {final_base}.l1.pkl + .l2.pkl + .l3.pkl + .l4.pkl + .l5.pkl")

    _text_benchmark_report(history)
    return history


def _parse_args():
    p = argparse.ArgumentParser(description="Hybrid text full simulation")
    p.add_argument("--corpus", required=True, help="Path to plain-text corpus file")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--log-every", type=int, default=1_000)
    p.add_argument("--chunk-size", type=int, default=120)
    p.add_argument("--warmup-chars", type=int, default=500)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dict", action="store_true", help="Enable dictionary and grammar validators")
    p.add_argument("--surface-mode", default="word", help="Text surface mode: word, ascii, or coarse")
    p.add_argument("--library", default=None, help="Base path to pre-built pattern library (no .pkl suffix)")
    p.add_argument("--checkpoint-dir", default=".", help="Directory for checkpoint files")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_text_simulation(
        corpus_path=args.corpus,
        total_steps=args.steps,
        log_every=args.log_every,
        chunk_size=args.chunk_size,
        warmup_chars=args.warmup_chars,
        num_workers=args.workers,
        use_dict=args.dict,
        surface_mode=args.surface_mode,
        library_path=args.library,
        checkpoint_dir=args.checkpoint_dir,
    )
