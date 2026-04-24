"""Wikipedia character-stream simulation for HPM v4."""
from __future__ import annotations

import os
from typing import Iterator, Any, Dict, List, Tuple

import numpy as np

# Vocabulary: newline=0, space=1, printable ASCII 33-126 = IDs 2-95
# Total: 96 symbols
VOCAB_SIZE = 96

_NEWLINE = '\n'
_NEWLINE_ID = 0
_ASCII_OFFSET = 31

def _build_maps():
    c2i = {_NEWLINE: _NEWLINE_ID}
    i2c = {_NEWLINE_ID: _NEWLINE}
    for code in range(32, 127):
        cid = code - _ASCII_OFFSET
        c2i[chr(code)] = cid
        i2c[cid] = chr(code)
    return c2i, i2c

_CHAR_TO_ID, _ID_TO_CHAR = _build_maps()

class WikipediaStream:
    """Streams character IDs from a text file, looping when exhausted."""
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._chars: list[int] = []
        self._pos: int = 0
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._filepath):
            self._chars = []
            return
        with open(self._filepath, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()
        self._chars = [_CHAR_TO_ID[ch] for ch in raw if ch in _CHAR_TO_ID]

    def __iter__(self) -> Iterator[int]:
        if not self._chars: return
        while True:
            yield self._chars[self._pos % len(self._chars)]
            self._pos += 1

    @staticmethod
    def char_to_id(ch: str) -> int: return _CHAR_TO_ID[ch]
    @staticmethod
    def id_to_char(i: int) -> str: return _ID_TO_CHAR[i]

def run_simulation(
    filepath: str,
    total_chars: int = 100_000,
    num_initial_patterns: int = 3,
    log_every: int = 1_000,
    num_workers: int = 4
) -> Tuple[Any, List[Dict]]:
    """
    Train an HPMAgent on a Wikipedia character stream.
    """
    from hpm_ai_v4.agents.agent import HPMAgent
    from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern

    stream = WikipediaStream(filepath)
    # HPMAgent now supports parallel evaluation internally via num_workers
    agent = HPMAgent(num_initial_patterns=0, obs_dim=VOCAB_SIZE, num_workers=num_workers)
    
    # Initialize patterns manually with latent_dim=4 for a fast balanced run
    agent.patterns = []
    for i in range(num_initial_patterns):
        p = HierarchicalPattern(pattern_id=i, latent_dim=4, obs_dim=VOCAB_SIZE)
        p.weight = 0.01
        agent.patterns.append(p)
            
    # Include flat pattern baseline
    flat_p = FlatPattern(num_initial_patterns, obs_dim=VOCAB_SIZE)
    flat_p.latent_dim = 4
    flat_p.weight = 0.95
    agent.patterns.append(flat_p)

    stream_iter = iter(stream)
    metrics_log = []

    print(f"--- Starting Parallel HPM Simulation ---")
    print(f"Config: {total_chars} steps, {num_workers} CPUs, {len(agent.patterns)} initial patterns.")
    
    try:
        for step in range(total_chars):
            try:
                char_id = next(stream_iter)
            except StopIteration:
                break
            
            # HPMAgent handles parallelization internally
            agent.perceive_and_learn(char_id)

            if step % log_every == 0 or step == total_chars - 1:
                # Compression on best pattern
                hier_patterns = [p for p in agent.patterns if p.complexity >= 2]
                comp = 0.0
                if hier_patterns:
                    best = max(hier_patterns, key=lambda p: p.weight)
                    comp = best.compression(agent.obs_buffer[-50:])
                
                print(f"[Step {step:6d}] patterns={len(agent.patterns):3d}  best_weight={max(p.weight for p in agent.patterns):.3f}  comp={comp:.4f}")
                metrics_log.append({'step': step, 'num_patterns': len(agent.patterns), 'compression': comp})
    finally:
        # Ensure the pool is closed even if interrupted
        agent._pool.close()

    return agent, metrics_log
