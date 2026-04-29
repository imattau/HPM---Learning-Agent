"""Text repair simulation for the stacked HPM text system."""
import argparse
import os
from typing import Any, Dict, List, Optional

from hpm_ai_v4.simulations.full_simulation import WikipediaStream
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.simulations.text_full_simulation import _ids_to_text, _text_metrics_snapshot
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary
from hpm_ai_v4.tools.text_signals import TextSignalExtractor


def _load_library(layered: LayeredAgent, library_path: Optional[str]) -> int:
    if not library_path or not os.path.exists(library_path):
        return 0
    if os.path.exists(library_path + ".l1.pkl"):
        return layered.load_bundle(library_path)
    from hpm_ai_v4.tools.serializer import PatternSerializer

    layered.l1.patterns = PatternSerializer.load(library_path)
    return 1 if layered.l1.patterns else 0


def _corrupt_text(text: str) -> str:
    """Create a deterministic noisy version of text for repair tests."""
    out: List[str] = []
    for idx, ch in enumerate(text):
        if ch.isalpha():
            if idx % 17 == 0:
                continue
            if idx % 11 == 0:
                base = ord("a") if ch.islower() else ord("A")
                out.append(chr(base + ((ord(ch.lower()) - ord("a") + 1) % 26)))
                continue
        elif ch.isdigit():
            if idx % 13 == 0:
                out.append(str((int(ch) + 1) % 10))
                continue
        elif ch.isspace():
            if idx % 19 == 0:
                continue
        elif idx % 23 == 0:
            continue
        out.append(ch)
    return "".join(out)


def _repair_benchmark_report(history: List[Dict[str, Any]]) -> None:
    if not history:
        print("No metrics recorded.")
        return

    final = history[-1]
    targets = [
        ("Repair agreement", "repair_agreement", 0.20),
        ("Repair improvement", "repair_improvement", 0.00),
        ("Plausibility", "plausibility", 0.35),
        ("L1 Compression MI", "compression_mi", 0.10),
        ("L2 Compression MI", "l2_compression_mi", 0.10),
        ("L3 Compression MI", "l3_compression_mi", 0.10),
        ("Population survived", "pop_size", 2),
    ]

    print("\n" + "=" * 60)
    print("TEXT REPAIR BENCHMARK REPORT")
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


def run_text_repair_simulation(
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
    """Run a minimal denoising / text repair benchmark."""
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    layered = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar, surface_mode=surface_mode)

    loaded = _load_library(layered, library_path)
    if loaded:
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
    text_signals = TextSignalExtractor()
    print(
        f"Starting text repair simulation: steps={total_steps} chunk={chunk_size} warmup={warmup_chars} "
        f"workers={num_workers} dict={use_dict}"
    )

    cursor = warmup_chars
    while cursor < warmup_chars + total_steps:
        target_text = corpus_text[cursor:cursor + chunk_size]
        if not target_text:
            break

        corrupted_text = _corrupt_text(target_text)
        repaired_text = layered.repair_text(
            corrupted_text=corrupted_text,
            target_text=target_text,
            mode="target",
            update_policy=True,
        )

        corruption_stats = layered.evaluate_generated_text(corrupted_text, target_text)
        repair_stats = layered.evaluate_generated_text(repaired_text, target_text)
        repair_stats["repair_improvement"] = repair_stats["token_agreement"] - corruption_stats["token_agreement"]
        repair_stats["corruption_agreement"] = corruption_stats["token_agreement"]
        repair_stats["corrupted_text"] = corrupted_text
        repair_signal = text_signals.analyze(
            repaired_text,
            context_texts=[corrupted_text, target_text],
            target_text=target_text,
            dictionary=layered.dictionary,
            grammar=layered.grammar,
        )
        repair_stats.update(repair_signal.to_dict())
        repair_stats["text_signal_score"] = repair_signal.combined_score()

        repair_stats.update(
            layered.observe_text(
                target_text,
                feedback_mode="hybrid",
                generated_text=repaired_text,
                self_feedback_weight=0.02,
                feedback_signal=repair_signal.to_dict(),
            )
        )

        snap = _text_metrics_snapshot(
            layered=layered,
            recent_chars=raw_ids[max(0, cursor - 240):cursor + len(target_text)],
            step=cursor,
            generated_text=repaired_text,
            target_text=target_text,
            target_stats=repair_stats,
        )
        snap["phase"] = "repair"
        snap["corrupted_text"] = corrupted_text
        snap["corruption_agreement"] = repair_stats["corruption_agreement"]
        snap["repair_agreement"] = repair_stats["token_agreement"]
        snap["repair_improvement"] = repair_stats["repair_improvement"]
        history.append(snap)

        if cursor % log_every == 0 or cursor == warmup_chars:
            print(
                f"[step {cursor:6d}] repair_agree={snap['repair_agreement']:.3f} "
                f"improve={snap['repair_improvement']:.3f} plaus={snap['plausibility']:.3f} "
                f"L1 mi={snap['compression_mi']:.3f} acc={snap['l1_accuracy']:.3f} "
                f"L2 acc={snap['l2_accuracy']:.3f} mi={snap['l2_compression_mi']:.3f} | "
                f"L3 acc={snap['l3_accuracy']:.3f} mi={snap['l3_compression_mi']:.3f} "
                f"stage={snap['dev_stage']}"
            )
            print(f"  target:    {target_text[:120]!r}")
            print(f"  corrupt:   {corrupted_text[:120]!r}")
            print(f"  repaired:  {repaired_text[:120]!r}")

        cursor += chunk_size

    os.makedirs(checkpoint_dir, exist_ok=True)
    final_base = os.path.join(checkpoint_dir, "final_repair_library")
    layered.save_bundle(final_base)
    print(f"Final repair library saved: {final_base}.l1.pkl + .l2.pkl + .l3.pkl + .l4.pkl + .l5.pkl")

    _repair_benchmark_report(history)
    return history


def _parse_args():
    p = argparse.ArgumentParser(description="Text repair simulation")
    p.add_argument("--corpus", required=True, help="Path to plain-text corpus file")
    p.add_argument("--total-steps", type=int, default=20_000)
    p.add_argument("--log-every", type=int, default=1_000)
    p.add_argument("--chunk-size", type=int, default=120)
    p.add_argument("--warmup-chars", type=int, default=500)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dict", action="store_true", help="Enable dictionary and grammar validators")
    p.add_argument("--library", default=None, help="Base path to pre-built pattern library (no .pkl suffix)")
    p.add_argument("--checkpoint-dir", default=".", help="Directory for checkpoint files")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_text_repair_simulation(
        corpus_path=args.corpus,
        total_steps=args.total_steps,
        log_every=args.log_every,
        chunk_size=args.chunk_size,
        warmup_chars=args.warmup_chars,
        num_workers=args.workers,
        use_dict=args.dict,
        library_path=args.library,
        checkpoint_dir=args.checkpoint_dir,
    )
