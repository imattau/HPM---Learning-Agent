# hpm_ai_v4/simulations/full_simulation.py
"""Full HPM AI simulation: all v4 features on a Wikipedia character stream."""
import argparse
import os
from typing import Iterator, List, Optional, Dict, Any

import numpy as np

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.io.adapters import CharClassAdapter
from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.serializer import PatternSerializer


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


def _metrics_snapshot(agent: HPMAgent, recent_chars: List[int], step: int) -> Dict[str, Any]:
    """Compute metrics over the recent character buffer."""
    snap: Dict[str, Any] = {'step': step}

    # --- Prediction accuracy ---
    correct = 0
    total = max(1, len(recent_chars) - 1)
    for i in range(len(recent_chars) - 1):
        context = recent_chars[max(0, i - 20):i]
        actual = recent_chars[i + 1]
        relevant = agent.reasoner.get_relevant_patterns(context, top_k=3)
        if relevant:
            pred_dist = agent.reasoner.compose_predictions(relevant, context)
            if int(np.argmax(pred_dist)) == actual:
                correct += 1
    snap['accuracy'] = correct / total

    # --- Compression MI ---
    top3 = sorted(agent.patterns, key=lambda p: p.weight, reverse=True)[:3]
    snap['compression_mi'] = float(np.mean([p.compression() for p in top3])) if top3 else 0.0

    # --- Population stats ---
    snap['pop_size'] = len(agent.patterns)
    snap['best_weight'] = float(max(p.weight for p in agent.patterns)) if agent.patterns else 0.0
    snap['dev_stage'] = agent.development.level
    snap['best_loss'] = float(min(p.running_loss for p in agent.patterns)) if agent.patterns else 0.0

    # --- Word completion (only if dictionary attached) ---
    if agent.dictionary:
        hits = 0
        # Extract up to 10 word prefixes of length 2-4 from recent chars
        prefixes = _extract_prefixes(recent_chars, n=10)
        for prefix_ids in prefixes:
            future = agent.reasoner.simulate_future(steps=8, top_k=3)
            word_ids = prefix_ids + future
            # Decode: char_id + 32 = ASCII
            word = ''.join(chr(v + 32) for v in word_ids if 0 <= v <= 94).strip()
            word = word.split()[0] if ' ' in word else word
            if word and agent.dictionary.contains(word.lower()):
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
        ('Prediction accuracy',  'accuracy',        0.50),
        ('Compression MI',       'compression_mi',  0.20),
        ('Population survived',  'pop_size',        2),     # > 1 pattern
    ]
    if final.get('word_completion') is not None:
        targets.append(('Word completion', 'word_completion', 0.30))

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
    print("Accuracy trajectory:")
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
    adapter = CharClassAdapter()  # maps 95 chars → 5 classes (obs_dim=5)

    agent = HPMAgent(
        obs_dim=5,
        num_initial_patterns=4,
        num_workers=num_workers,
        dictionary=dictionary,
    )

    # Equal initial weights: hier patterns compete fairly against flat baseline
    agent.patterns = []
    for i in range(4):
        p = HierarchicalPattern(i, latent_dim=2, obs_dim=5)
        p.weight = 0.15
        agent.patterns.append(p)
    for i in range(4, 6):
        p = FlatPattern(i, obs_dim=5)
        p.weight = 0.1
        agent.patterns.append(p)

    if library_path and os.path.exists(library_path):
        n = agent.load_library(library_path, reset_weights=True)
        print(f"Loaded {n} patterns from {library_path}")

    stream = WikipediaStream(corpus_path)
    stream_iter = iter(stream)

    history: List[Dict[str, Any]] = []
    recent_chars: List[int] = []
    accuracy_buffer: List[int] = []  # actual chars for accuracy computation

    print(f"Starting simulation: steps={total_steps} log_every={log_every} "
          f"workers={num_workers} dict={use_dict}")

    for step in range(total_steps):
        raw_id = next(stream_iter)
        char_id = adapter.encode(raw_id)  # 0–94 → 0–4 char class
        accuracy_buffer.append(char_id)
        if len(accuracy_buffer) > log_every + 21:
            accuracy_buffer = accuracy_buffer[-(log_every + 21):]

        agent.perceive_and_learn(char_id)

        recent_chars.append(char_id)
        if len(recent_chars) > log_every:
            recent_chars = recent_chars[-log_every:]

        if step % log_every == 0 and step > 0:
            snap = _metrics_snapshot(agent, list(accuracy_buffer), step)
            history.append(snap)
            wc = f"{snap['word_completion']:.3f}" if snap['word_completion'] is not None else 'n/a'
            print(
                f"[step {step:6d}] acc={snap['accuracy']:.3f} "
                f"mi={snap['compression_mi']:.3f} "
                f"wc={wc} "
                f"pop={snap['pop_size']} "
                f"stage={snap['dev_stage']} "
                f"loss={snap['best_loss']:.3f}"
            )

        if step % 10_000 == 0 and step > 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pkl")
            PatternSerializer.save(agent.patterns, ckpt_path)
            print(f"  [checkpoint saved: {ckpt_path}]")

    final_path = os.path.join(checkpoint_dir, "final_library.pkl")
    PatternSerializer.save(agent.patterns, final_path)
    print(f"Final library saved: {final_path}")

    _benchmark_report(history)
    return history


def _parse_args():
    p = argparse.ArgumentParser(description="Full HPM AI simulation")
    p.add_argument('--corpus', required=True, help='Path to plain-text corpus file')
    p.add_argument('--steps', type=int, default=100_000)
    p.add_argument('--log-every', type=int, default=1_000)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--dict', action='store_true', help='Enable NLTKWordList dictionary')
    p.add_argument('--library', default=None, help='Path to pre-built pattern library (.pkl)')
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
