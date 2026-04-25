#!/usr/bin/env python3
"""
Integrated Hierarchical Topic Distillation.
Fetches a Wikipedia topic, runs a 3-level HPM stack, and saves distilled winners.

Usage:
    PYTHONPATH=. python3 hpm_ai_v4/simulations/experiment_topic_distillation.py \
        --topic "Quantum Mechanics" --steps 10000 --output quantum_library.pkl
"""
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from hpm_ai_v4.simulations.wikipedia_sim import (
    _make_population, _update_level, 
    PatternField, ParallelPatternPool, WikipediaStream
)
from hpm_ai_v4.simulations.build_library import fetch_wikipedia_content, compute_density
from hpm_ai_v4.tools.serializer import PatternSerializer
from hpm_ai_v4.io.adapters import CharClassAdapter


def run_distillation(topic: str, total_steps: int, output_path: str, 
                     input_path: str = None, num_workers: int = 2):
    print(f"\n[distill] Starting Hierarchical Distillation for topic: {topic!r}")
    if input_path:
        print(f"[distill] Loading innate priors from: {input_path}")
    
    # 1. Fetch Content
    corpus = fetch_wikipedia_content(topic)
    if not corpus:
        print("[error] Could not build corpus. Aborting.")
        return
    
    # Save a local temp file for WikipediaStream
    temp_corpus = f"temp_{topic.replace(' ', '_')}.txt"
    with open(temp_corpus, 'w', encoding='utf-8') as f:
        f.write(corpus)

    adapter = CharClassAdapter()
    stream = WikipediaStream(temp_corpus, adapter)
    stream_iter = iter(stream)

    # 2. Initialize 3-Level Stack
    # Load from library if available, otherwise random
    loaded_patterns = []
    if input_path and os.path.exists(input_path):
        loaded_patterns = PatternSerializer.load(input_path)
    
    def get_population(lvl_name, default_n, latent_dim, obs_dim):
        lvl_pats = [p for p in loaded_patterns if getattr(p, 'level', None) == lvl_name]
        if lvl_pats:
            print(f"[init] {lvl_name}: Loaded {len(lvl_pats)} patterns from library.")
            # Tag as loaded
            for p in lvl_pats: 
                p.provenance = getattr(p, 'provenance', 'innate')
                p.is_loaded = True
            # Normalise weights
            w = 1.0 / len(lvl_pats)
            for p in lvl_pats: p.weight = w
            # Refresh caches
            for p in lvl_pats: p._refresh_log_cache()
            return lvl_pats
        else:
            print(f"[init] {lvl_name}: Initializing {default_n} random patterns.")
            pats = _make_population(default_n, latent_dim=latent_dim, obs_dim=obs_dim)
            for p in pats:
                p.provenance = f'init:{topic}'
                p.is_loaded = False
            return pats

    L1 = get_population('L1', 10, latent_dim=2, obs_dim=adapter.obs_dim)
    L2 = get_population('L2', 6,  latent_dim=2, obs_dim=2)
    L3 = get_population('L3', 4,  latent_dim=2, obs_dim=2)

    f1, f2, f3 = PatternField(), PatternField(), PatternField()
    buf1, buf2, buf3 = [], [], []

    pool = ParallelPatternPool(num_workers=num_workers)
    history = []
    log_every = 500

    print(f"[init] Stacking 3 levels (L1, L2, L3) | {total_steps} steps")

    try:
        for step in range(total_steps):
            try:
                class_id = next(stream_iter)
            except StopIteration:
                print("[info] End of corpus reached. Stopping.")
                break

            # Level 1 (Character)
            _update_level(L1, class_id, buf1, f1, step, pool)

            # Level 2 (Local Structure)
            best_L1 = max(L1, key=lambda p: p.weight)
            l1_state = best_L1.get_top_state(buf1[-20:])
            _update_level(L2, l1_state, buf2, f2, step, pool)

            # Level 3 (Meta Structure)
            best_L2 = max(L2, key=lambda p: p.weight)
            l2_state = best_L2.get_top_state(buf2[-20:])
            _update_level(L3, l2_state, buf3, f3, step, pool)

            if step > 0 and step % log_every == 0:
                best_l1_loss = min(p.running_loss for p in L1)
                best_l2_loss = min(p.running_loss for p in L2)
                best_l3_loss = min(p.running_loss for p in L3)
                
                print(f"[step {step:6d}] L1_loss={best_l1_loss:.3f} | "
                      f"L2_loss={best_l2_loss:.3f} | L3_loss={best_l3_loss:.3f}")
                history.append((step, best_l1_loss, best_l2_loss, best_l3_loss))

    finally:
        pool.close()
        if os.path.exists(temp_corpus):
            os.remove(temp_corpus)

    # 3. Analyze and Plot
    _plot_history(history, topic)

    # 4. Distill and Save
    all_kept = []
    print("\n[analysis] Distillation Winners:")
    for lvl_name, population, buffer in [('L1', L1, buf1), ('L2', L2, buf2), ('L3', L3, buf3)]:
        field_freq = {p.id: p.weight for p in population}
        winners = []
        loaded_count = 0
        new_count = 0
        for p in population:
            d = compute_density(p, buffer, field_freq)
            if p.weight > 0.1:
                p.level = lvl_name
                p.density_at_save = d
                # Update provenance if it was a new discovery/recombination in this session
                if not getattr(p, 'is_loaded', False):
                    p.provenance = f"new:{topic}"
                    new_count += 1
                else:
                    loaded_count += 1
                winners.append(p)
        
        print(f"  {lvl_name}: {len(winners)} total (Loaded: {loaded_count}, New: {new_count})")
        all_kept.extend(winners)

    if all_kept:
        PatternSerializer.save(all_kept, output_path)
        print(f"[save] Multi-level library saved to: {output_path}")
    else:
        print("[warn] No patterns passed distillation threshold.")


def _plot_history(history, topic):
    if not history: return
    steps, l1, l2, l3 = zip(*history)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, l1, label='L1 (Char) Loss')
    plt.plot(steps, l2, label='L2 (Local) Loss')
    plt.plot(steps, l3, label='L3 (Meta) Loss')
    plt.title(f'Hierarchical Learning Progress: {topic}')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_name = f"distill_{topic.replace(' ', '_')}.png"
    plt.savefig(plot_name)
    print(f"[plot] Analysis saved to {plot_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated Hierarchical Topic Distillation")
    parser.add_argument('--topic', required=True, help='Wikipedia topic')
    parser.add_argument('--steps', type=int, default=10000, help='Total simulation steps')
    parser.add_argument('--output', required=True, help='Path to save .pkl library')
    parser.add_argument('--input-library', help='Path to load existing .pkl library')
    parser.add_argument('--workers', type=int, default=2, help='Parallel workers')
    args = parser.parse_args()

    run_distillation(args.topic, args.steps, args.output, 
                     input_path=args.input_library, num_workers=args.workers)
