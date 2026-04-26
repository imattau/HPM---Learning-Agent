"""Held-out text generalization simulation for the stacked HPM text system."""
import argparse
import os
from statistics import mean
from typing import Any, Dict, List, Optional

from hpm_ai_v4.simulations.full_simulation import WikipediaStream
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.simulations.text_full_simulation import _ids_to_text, _text_metrics_snapshot
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary
from hpm_ai_v4.tools.serializer import PatternSerializer


def _phase_report(label: str, final: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"{label.upper()} REPORT")
    print("=" * 60)
    print(f"target_agreement={final.get('target_agreement', 0.0):.3f} "
          f"plausibility={final.get('plausibility', 0.0):.3f} "
          f"L1_mi={final.get('compression_mi', 0.0):.3f} "
          f"L2_mi={final.get('l2_compression_mi', 0.0):.3f} "
          f"L3_mi={final.get('l3_compression_mi', 0.0):.3f}")


def _validation_summary(history: List[Dict[str, Any]]) -> Dict[str, float]:
    val = [snap for snap in history if snap.get("phase") == "validation"]
    if not val:
        return {
            "target_agreement": 0.0,
            "plausibility": 0.0,
            "compression_mi": 0.0,
            "l2_compression_mi": 0.0,
            "l3_compression_mi": 0.0,
            "l1_accuracy": 0.0,
            "l2_accuracy": 0.0,
            "l3_accuracy": 0.0,
        }

    return {
        "target_agreement": float(mean(s["target_agreement"] for s in val)),
        "plausibility": float(mean(s["plausibility"] for s in val)),
        "compression_mi": float(mean(s["compression_mi"] for s in val)),
        "l2_compression_mi": float(mean(s["l2_compression_mi"] for s in val)),
        "l3_compression_mi": float(mean(s["l3_compression_mi"] for s in val)),
        "l1_accuracy": float(mean(s["l1_accuracy"] for s in val)),
        "l2_accuracy": float(mean(s["l2_accuracy"] for s in val)),
        "l3_accuracy": float(mean(s["l3_accuracy"] for s in val)),
    }


def run_text_generalization_simulation(
    corpus_path: str,
    train_steps: int = 20_000,
    validation_steps: int = 4_000,
    log_every: int = 1_000,
    chunk_size: int = 120,
    prompt_size: int = 200,
    warmup_chars: int = 500,
    num_workers: int = 1,
    use_dict: bool = True,
    library_path: Optional[str] = None,
    checkpoint_dir: str = ".",
) -> List[Dict[str, Any]]:
    """Train on one slice of text, then validate on a held-out continuation."""
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    layered = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)

    if library_path and os.path.exists(library_path + ".l1.pkl"):
        layered.load_bundle(library_path)
        print(f"Loaded library from {library_path}")

    stream = WikipediaStream(corpus_path)
    stream_iter = iter(stream)
    needed = warmup_chars + train_steps + validation_steps + chunk_size + prompt_size
    raw_ids = [next(stream_iter) for _ in range(needed)]
    corpus_text = _ids_to_text(raw_ids)

    train_text = corpus_text[: warmup_chars + train_steps]
    validation_text = corpus_text[warmup_chars + train_steps : warmup_chars + train_steps + validation_steps + chunk_size]

    warmup_text = train_text[:warmup_chars]
    if warmup_text:
        layered.observe_text(warmup_text, feedback_mode="target")

    train_tail = train_text[warmup_chars:]
    for ch in train_tail:
        layered.perceive(94 if ch == "\n" else ord(ch) - 32)

    history: List[Dict[str, Any]] = []
    train_snapshot = _text_metrics_snapshot(
        layered=layered,
        recent_chars=raw_ids[: len(train_text)],
        step=len(train_text),
        generated_text="",
        target_text="",
        target_stats={"token_agreement": 0.0, "plausibility": 0.0, "target_chars": len(train_text), "self_chars": 0},
    )
    train_snapshot["phase"] = "train"
    train_snapshot["l4_compression_mi"] = layered.l4_metrics()["mi"]
    train_snapshot["l5_compression_mi"] = layered.l5_metrics()["mi"]
    history.append(train_snapshot)

    print(
        f"Starting generalization simulation: train_steps={train_steps} validation_steps={validation_steps} "
        f"chunk={chunk_size} warmup={warmup_chars} prompt={prompt_size} workers={num_workers} dict={use_dict}"
    )

    cursor = 0
    while cursor < validation_steps:
        target_text = validation_text[cursor:cursor + chunk_size]
        if not target_text:
            break

        seed_start = max(0, cursor - prompt_size)
        seed_text = validation_text[seed_start:cursor]
        planned_text = layered.plan_text_continuation(
            target_text=target_text,
            seed_text=seed_text[-200:] if seed_text else None,
            horizon=max(8, min(80, len(target_text) // 2)),
            strategy="beam",
            lookback=240,
        )
        if use_dict:
            generated_text = layered.generate_constrained_text(
                steps=max(8, min(80, len(target_text) // 2)),
                seed_text=seed_text[-200:] if seed_text else None,
                target_text=target_text,
                mode="target",
                include_seed=False,
                allowed_words=set(layered.dictionary.words) if layered.dictionary else None,
                strict_dictionary=True,
                strict_grammar=True,
                update_policy=False,
            )
        else:
            generated_text = layered.generate_text(
                steps=max(8, min(80, len(target_text) // 2)),
                seed_text=seed_text[-200:] if seed_text else None,
                target_text=target_text,
                mode="target",
                include_seed=False,
                update_policy=False,
            )

        planned_stats = layered.evaluate_generated_text(planned_text, target_text) if planned_text else {"token_agreement": 0.0, "plausibility": 0.0}
        target_stats = layered.evaluate_generated_text(generated_text, target_text)
        target_stats["planned_agreement"] = planned_stats["token_agreement"]
        target_stats["planned_plausibility"] = planned_stats["plausibility"]

        snap = _text_metrics_snapshot(
            layered=layered,
            recent_chars=raw_ids[len(train_text) + cursor - prompt_size: len(train_text) + cursor + len(target_text)],
            step=len(train_text) + cursor,
            generated_text=generated_text,
            target_text=target_text,
            target_stats=target_stats,
        )
        snap["phase"] = "validation"
        snap["planned_agreement"] = planned_stats["token_agreement"]
        snap["planned_plausibility"] = planned_stats["plausibility"]
        snap["l4_compression_mi"] = layered.l4_metrics()["mi"]
        snap["l5_compression_mi"] = layered.l5_metrics()["mi"]
        history.append(snap)

        if cursor % log_every == 0:
            print(
                f"[val {cursor:6d}] gen_agree={snap['target_agreement']:.3f} plaus={snap['plausibility']:.3f} "
                f"L1 mi={snap['compression_mi']:.3f} L2 mi={snap['l2_compression_mi']:.3f} "
                f"L3 mi={snap['l3_compression_mi']:.3f}"
            )
            print(f"  target:  {target_text[:120]!r}")
            print(f"  planned: {planned_text[:120]!r}")
            print(f"  generated: {generated_text[:120]!r}")

        cursor += chunk_size

    os.makedirs(checkpoint_dir, exist_ok=True)
    final_base = os.path.join(checkpoint_dir, "final_generalization_library")
    layered.save_bundle(final_base)
    print(f"Final generalization library saved: {final_base}.l1.pkl + .l2.pkl + .l3.pkl + .l4.pkl + .l5.pkl")

    train_final = history[0]
    val_final = _validation_summary(history)

    _phase_report("train", train_final)
    _phase_report("validation (mean)", val_final)
    return history


def _parse_args():
    p = argparse.ArgumentParser(description="Held-out text generalization simulation")
    p.add_argument("--corpus", required=True, help="Path to plain-text corpus file")
    p.add_argument("--train-steps", type=int, default=20_000)
    p.add_argument("--validation-steps", type=int, default=4_000)
    p.add_argument("--log-every", type=int, default=1_000)
    p.add_argument("--chunk-size", type=int, default=120)
    p.add_argument("--prompt-size", type=int, default=200)
    p.add_argument("--warmup-chars", type=int, default=500)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dict", action="store_true", help="Enable dictionary and grammar validators")
    p.add_argument("--library", default=None, help="Base path to pre-built pattern library (no .pkl suffix)")
    p.add_argument("--checkpoint-dir", default=".", help="Directory for checkpoint files")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_text_generalization_simulation(
        corpus_path=args.corpus,
        train_steps=args.train_steps,
        validation_steps=args.validation_steps,
        log_every=args.log_every,
        chunk_size=args.chunk_size,
        prompt_size=args.prompt_size,
        warmup_chars=args.warmup_chars,
        num_workers=args.workers,
        use_dict=args.dict,
        library_path=args.library,
        checkpoint_dir=args.checkpoint_dir,
    )
