# hpm_ai_v4/simulations/full_simulation.py
"""Full HPM AI simulation: all v4 features on a Wikipedia character stream."""
import argparse
import os
from typing import Iterator, List, Optional, Dict, Any

import numpy as np

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.serializer import PatternSerializer
from hpm_ai_v4.simulations.layered_agent import LayeredAgent


class WikipediaStream:
    """Yields char IDs (0–94) from a text file, looping on exhaustion."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def __iter__(self) -> Iterator[int]:
        while True:
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for ch in f.read():
                    if ch == '\n':
                        yield 94
                    else:
                        code = ord(ch)
                        if 32 <= code <= 126:
                            yield code - 32


def _metrics_snapshot(layered: LayeredAgent, recent_chars: List[int], step: int) -> Dict[str, Any]:
    """Compute metrics over the recent character buffer."""
    snap: Dict[str, Any] = {'step': step}

    # L1 metrics
    m1 = layered.l1_metrics()
    snap['compression_mi'] = m1['mi']
    snap['pop_size'] = m1['pop_size']
    snap['best_weight'] = m1['best_weight']
    snap['dev_stage'] = m1['stage']
    snap['best_loss'] = float(min(p.running_loss for p in layered.l1.patterns)) if layered.l1.patterns else 0.0

    # L1 accuracy (class level)
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
    snap['accuracy'] = correct / total

    # L2 accuracy (raw char, sampled over last 200 for speed)
    sample = recent_chars[-200:] if len(recent_chars) > 200 else recent_chars
    m2 = layered.l2_metrics(sample)
    snap['l2_accuracy'] = m2['accuracy']
    snap['l2_pop_size'] = m2['pop_size']
    
    # --- Word completion (only if dictionary attached to L2) ---
    if layered.l2.dictionary:
        hits = 0
        prefixes = _extract_prefixes(recent_chars, n=10)
        for prefix_ids in prefixes:
            future = layered.l2.reasoner.simulate_future(steps=8, top_k=3)
            word_ids = prefix_ids + future
            word = ''.join(chr(v + 32) for v in word_ids if 0 <= v <= 94).strip()
            word = word.split()[0] if ' ' in word else word
            if word and layered.l2.dictionary.contains(word.lower()):
                hits += 1
        snap['word_completion'] = hits / max(1, len(prefixes))
    else:
        snap['word_completion'] = None

    return snap


def _extract_prefixes(chars: List[int], n: int = 10) -> List[List[int]]:
    """Extract up to n word prefixes (length 2-4) from char ID sequence."""
    # Space = char_id 0 (ASCII 32 - 32 = 0)
    space_id = 0
    prefixes = []
    i = 0
    while i < len(chars) and len(prefixes) < n:
        if chars[i] == space_id and i + 1 < len(chars):
            # Start of a word — take 2-4 chars
            end = min(i + 5, len(chars))
            word_chars = [c for c in chars[i+1:end] if c != space_id]
            if 2 <= len(word_chars) <= 4:
                prefixes.append(word_chars)
        i += 1
    return prefixes


def _benchmark_report(history: List[Dict[str, Any]]) -> None:
    """Print pass/fail benchmark summary from metric history."""
    if not history:
        print("No metrics recorded.")
        return

    final = history[-1]
    targets = [
        ('L1 Prediction accuracy', 'accuracy',        0.50),
        ('L1 Compression MI',       'compression_mi',  0.10),
        ('L2 Prediction accuracy', 'l2_accuracy',     0.10),
        ('Population survived',    'pop_size',        2),
    ]
    if final.get('word_completion') is not None:
        targets.append(('Word completion', 'word_completion', 0.20))

    print("\n" + "=" * 60)
    print("BENCHMARK REPORT")
    print("=" * 60)
    print(f"{'Metric':<28} {'Target':>8} {'Final':>8} {'':>6}")
    print("-" * 60)
    all_pass = True
    for label, key, target in targets:
        val = final.get(key, 0.0)
        passed = val > target
        if not passed:
            all_pass = False
        mark = 'PASS' if passed else 'FAIL'
        print(f"{label:<28} {target:>8.2f} {val:>8.2f} {mark:>6}")

    print("=" * 60)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()

    # Trajectory
    print("Accuracy trajectory (L1):")
    for snap in history[::max(1, len(history)//10)]:
        bar = '#' * int(snap.get('accuracy', 0) * 40)
        print(f"  step {snap['step']:6d}: {snap.get('accuracy', 0):.3f} {bar}")
    print()


def run_simulation(
    corpus_path: str,
    total_steps: int = 100_000,
    log_every: int = 1_000,
    num_workers: int = 1,
    use_dict: bool = False,
    library_path: Optional[str] = None,
    checkpoint_dir: str = '.',
) -> List[Dict[str, Any]]:
    """Run the full HPM AI simulation. Returns metric history."""

    dictionary = NLTKWordList() if use_dict else None
    layered = LayeredAgent(num_workers=num_workers)
    
    if use_dict:
        layered.l2.dictionary = dictionary

    if library_path and os.path.exists(library_path + ".l1.pkl"):
        layered.l1.patterns = PatternSerializer.load(library_path + ".l1.pkl")
        layered.l2.patterns = PatternSerializer.load(library_path + ".l2.pkl")
        print(f"Loaded library from {library_path}")

    stream = WikipediaStream(corpus_path)
    stream_iter = iter(stream)

    history: List[Dict[str, Any]] = []
    accuracy_buffer: List[int] = []  # actual chars for accuracy computation

    print(f"Starting simulation: steps={total_steps} log_every={log_every} "
          f"workers={num_workers} dict={use_dict}")

    for step in range(total_steps):
        raw_id = next(stream_iter)
        accuracy_buffer.append(raw_id)
        if len(accuracy_buffer) > log_every + 21:
            accuracy_buffer = accuracy_buffer[-(log_every + 21):]

        layered.perceive(raw_id)

        if step % log_every == 0 and step > 0:
            snap = _metrics_snapshot(layered, list(accuracy_buffer), step)
            history.append(snap)
            wc = f"{snap['word_completion']:.3f}" if snap['word_completion'] is not None else 'n/a'
            print(
                f"[step {step:6d}] "
                f"L1 acc={snap['accuracy']:.3f} mi={snap['compression_mi']:.3f} "
                f"pop={snap['pop_size']} stage={snap['dev_stage']} | "
                f"L2 acc={snap['l2_accuracy']:.3f} pop={snap['l2_pop_size']}"
            )

        if step % 10_000 == 0 and step > 0:
            base = os.path.join(checkpoint_dir, f"checkpoint_{step}")
            PatternSerializer.save(layered.l1.patterns, base + ".l1.pkl")
            PatternSerializer.save(layered.l2.patterns, base + ".l2.pkl")
            print(f"  [checkpoint saved: {base}.l1.pkl + .l2.pkl]")

    base_final = os.path.join(checkpoint_dir, "final_library")
    PatternSerializer.save(layered.l1.patterns, base_final + ".l1.pkl")
    PatternSerializer.save(layered.l2.patterns, base_final + ".l2.pkl")
    print(f"Final library saved: {base_final}.l1.pkl + .l2.pkl")

    _benchmark_report(history)
    return history


def _parse_args():
    p = argparse.ArgumentParser(description="Full HPM AI simulation")
    p.add_argument('--corpus', required=True, help='Path to plain-text corpus file')
    p.add_argument('--steps', type=int, default=100_000)
    p.add_argument('--log-every', type=int, default=1_000)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--dict', action='store_true', help='Enable NLTKWordList dictionary')
    p.add_argument('--library', default=None, help='Base path to pre-built pattern library (no .pkl suffix)')
    p.add_argument('--checkpoint-dir', default='.', help='Directory for checkpoint files')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_simulation(
        corpus_path=args.corpus,
        total_steps=args.steps,
        log_every=args.log_every,
        num_workers=args.workers,
        use_dict=args.dict,
        library_path=args.library,
        checkpoint_dir=args.checkpoint_dir,
    )
