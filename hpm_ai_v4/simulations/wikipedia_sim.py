# hpm_ai_v4/simulations/wikipedia_sim.py
from typing import Iterator, Dict, Any, List
from hpm_ai_v4.io.adapters import CharClassAdapter


class WikipediaStream:
    """Reads a plain-text file, converts chars to class IDs, loops on exhaustion."""

    def __init__(self, filepath: str, adapter: CharClassAdapter):
        self.filepath = filepath
        self.adapter = adapter

    def _char_to_class(self, ch: str):
        if ch == '\n':
            return self.adapter.encode(-22)
        code = ord(ch)
        if 32 <= code <= 126:
            return self.adapter.encode(code - 32)
        return None  # skip

    def __iter__(self) -> Iterator[int]:
        while True:
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for ch in f.read():
                    cls = self._char_to_class(ch)
                    if cls is not None:
                        yield cls

import numpy as np
from hpm_ai_v4.pattern import HierarchicalPattern
from hpm_ai_v4.field import PatternField
from hpm_ai_v4.operators.dynamics import compute_conflict_matrix, meta_pattern_update, recombine
from hpm_ai_v4.operators.parallel import ParallelPatternPool


def _make_population(n: int, latent_dim: int, obs_dim: int) -> List[HierarchicalPattern]:
    patterns = []
    for i in range(n):
        p = HierarchicalPattern(pattern_id=i, latent_dim=latent_dim, obs_dim=obs_dim)
        p.weight = 1.0 / n
        patterns.append(p)
    return patterns


def _update_level(patterns, obs, buffer, field, step, pool: ParallelPatternPool, recombine_every=50):
    """Single-step observe + adapt + replicator for one level (Parallel)."""
    buffer.append(obs)
    if len(buffer) > 100:
        buffer[:] = buffer[-100:]

    field_freq = field.update(patterns)
    
    worker_params = {
        'learning_rate': 0.02,
        'lambda_l': 0.1,
        'adapt_window': 30,
        'beta_aff': 0.4,
        'gamma_soc': 0.3,
        'external_soc': 0.5,
    }

    results = pool.map_patterns(patterns, buffer, field_freq, worker_params)

    # Write updated state back
    result_by_id = {r['pattern_id']: r for r in results}
    totals = {}
    for p in patterns:
        r = result_by_id[p.id]
        if p.complexity >= 1 or p.latent_dim > 1:
            p.A = r['A']
            p.B = r['B']
            p.pi = r['pi']
            p._refresh_log_cache()
        else:
            p.B = r['B']
        p.running_loss = r['running_loss']
        totals[p.id] = r['total_score']

    k_mat = compute_conflict_matrix(patterns)
    meta_pattern_update(patterns, totals, eta=0.1, beta_c=0.03,
                        k_matrix=k_mat, decay=0.005)

    # Recombination
    if step > 0 and step % recombine_every == 0 and len(patterns) >= 2:
        weights = np.array([p.weight for p in patterns])
        if weights.sum() > 0:
            probs = weights / weights.sum()
            idxs = np.random.choice(len(patterns), size=2, p=probs, replace=False)
            child = recombine(patterns[idxs[0]], patterns[idxs[1]])
            if child is not None:
                child.id = max(p.id for p in patterns) + 1
                child.weight = 0.05
                patterns.append(child)

    patterns[:] = [p for p in patterns if p.weight > 1e-4]


def run_simulation(corpus_path: str, total_steps: int = 100_000,
                   log_every: int = 1_000, num_workers: int = 1) -> Dict[str, Any]:
    adapter = CharClassAdapter()
    stream = WikipediaStream(corpus_path, adapter)
    stream_iter = iter(stream)

    L1 = _make_population(10, latent_dim=2, obs_dim=5)
    L2 = _make_population(5,  latent_dim=2, obs_dim=2)
    L3 = _make_population(3,  latent_dim=2, obs_dim=2)

    f1, f2, f3 = PatternField(), PatternField(), PatternField()
    buf1, buf2, buf3 = [], [], []

    pool = ParallelPatternPool(num_workers=num_workers)

    try:
        for step in range(total_steps):
            class_id = next(stream_iter)

            # Level 1
            _update_level(L1, class_id, buf1, f1, step, pool)

            # Extract L1 latent → L2 input
            best_L1 = max(L1, key=lambda p: p.weight)
            l1_state = best_L1.get_top_state(buf1[-20:])

            # Level 2
            _update_level(L2, l1_state, buf2, f2, step, pool)

            # Extract L2 latent → L3 input
            best_L2 = max(L2, key=lambda p: p.weight)
            l2_state = best_L2.get_top_state(buf2[-20:])

            # Level 3
            _update_level(L3, l2_state, buf3, f3, step, pool)

            if step % log_every == 0:
                best_l1_loss = min(p.running_loss for p in L1)
                print(f"[step {step:6d}] L1 patterns={len(L1)} best_loss={best_l1_loss:.3f} "
                      f"L2 patterns={len(L2)} L3 patterns={len(L3)}")
    finally:
        pool.close()

    return {'L1_patterns': L1, 'L2_patterns': L2, 'L3_patterns': L3,
            'buf1': buf1, 'buf2': buf2, 'buf3': buf3}
