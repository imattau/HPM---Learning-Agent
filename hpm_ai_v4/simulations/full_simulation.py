# hpm_ai_v4/simulations/full_simulation.py
"""Full HPM AI simulation: all v4 features on a Wikipedia character stream."""
import argparse
import os
from typing import Iterator, List, Optional, Dict, Any

import numpy as np

from hpm_ai_v4.agents.agent import HPMAgent
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
    print(f"corpus={args.corpus} steps={args.steps} workers={args.workers}")
