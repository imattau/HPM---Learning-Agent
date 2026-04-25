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
