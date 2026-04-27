#!/usr/bin/env python3
"""
Build a large NLP pattern library (2000+ patterns) by training many agents
on diverse text chunks from HuggingFace datasets and aggregating.

Usage:
    python3 -m hpm_ai_v4.simulations.build_large_nlp_library \
        --output library_bootstrap/nlp_large.pkl \
        --target 2000 \
        --steps-per-chunk 8000 \
        --workers 4
"""
import argparse
import os
import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import List, Optional

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.io.adapters import CharClassAdapter
from hpm_ai_v4.evaluators.metrics import epistemic_score, affective_score, social_score, pattern_density
from hpm_ai_v4.tools.serializer import PatternSerializer
from hpm_ai_v4.pattern import HierarchicalPattern


# ---------------------------------------------------------------------------
# Dataset sources
# ---------------------------------------------------------------------------

def load_nltk_chunks(target_chars: int = 5_000_000) -> List[str]:
    """Load diverse text from NLTK corpora (no download/auth required)."""
    import nltk
    chunks = []
    total = 0
    chunk_size = 5000

    corpora = [
        ("brown", None),
        ("gutenberg", None),
        ("reuters", None),
        ("webtext", None),
        ("inaugural", None),
    ]

    for corpus_name, _ in corpora:
        if total >= target_chars:
            break
        try:
            corpus = getattr(nltk.corpus, corpus_name)
            try:
                text = corpus.raw()
            except Exception:
                words = corpus.words()
                text = " ".join(words)
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                if len(chunk) < 100:
                    continue
                chunks.append(chunk)
                total += len(chunk)
                if total >= target_chars:
                    break
            print(f"[data] nltk.{corpus_name}: {total:,} chars accumulated")
        except Exception as e:
            try:
                nltk.download(corpus_name, quiet=True)
                corpus = getattr(nltk.corpus, corpus_name)
                try:
                    text = corpus.raw()
                except Exception:
                    text = " ".join(corpus.words())
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size]
                    if len(chunk) >= 100:
                        chunks.append(chunk)
                        total += len(chunk)
                        if total >= target_chars:
                            break
                print(f"[data] nltk.{corpus_name} (downloaded): {total:,} chars accumulated")
            except Exception as e2:
                print(f"[warn] Could not load nltk.{corpus_name}: {e2}")

    if not chunks:
        print("[fallback] Using local wiki_sample.txt")
        local = "hpm_ai_v4/simulations/data/wiki_sample.txt"
        if os.path.exists(local):
            with open(local, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    print(f"[data] {len(chunks)} text chunks ready ({total:,} chars)")
    return chunks


def load_hf_chunks(target_chars: int = 5_000_000) -> List[str]:
    """Load text chunks using NLTK corpora (no network auth required)."""
    return load_nltk_chunks(target_chars)


# ---------------------------------------------------------------------------
# Single-chunk training (runs in worker process)
# ---------------------------------------------------------------------------

def _train_chunk(args):
    chunk_text, steps, min_density, chunk_idx = args
    adapter = CharClassAdapter()
    tokens = [adapter.encode_char(ch) for ch in chunk_text if 32 <= ord(ch) <= 127 or ch == '\n']
    if len(tokens) < 50:
        return []

    agent = HPMAgent(num_initial_patterns=20, obs_dim=5)
    n = len(tokens)
    for step in range(steps):
        obs = tokens[step % n]
        agent.perceive_and_learn(obs)

    field_freq = {p.id: p.weight for p in agent.patterns}
    kept = []
    for p in agent.patterns:
        if p.latent_dim <= 1:
            continue  # skip FlatPatterns
        ep = epistemic_score(p)
        aff = affective_score(p, agent.obs_buffer)
        soc = social_score(p, field_freq)
        field_infl = 0.2 * soc
        d = pattern_density(p, agent.obs_buffer, [ep, aff, soc, field_infl])
        if d >= min_density and p.weight > 0.01:
            p.source_corpus = f"chunk_{chunk_idx}"
            p.density_at_save = float(d)
            kept.append(p)

    return kept


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _pattern_fingerprint(p: HierarchicalPattern) -> np.ndarray:
    return np.concatenate([p.A.flatten(), p.B.flatten(), p.pi.flatten()])


def deduplicate(patterns: List[HierarchicalPattern], sim_threshold: float = 0.97) -> List[HierarchicalPattern]:
    """Remove near-duplicate patterns by cosine similarity of parameter vectors."""
    if not patterns:
        return []
    fps = [_pattern_fingerprint(p) for p in patterns]
    norms = [np.linalg.norm(fp) + 1e-12 for fp in fps]
    kept_idx = []
    for i, (fp_i, norm_i) in enumerate(zip(fps, norms)):
        duplicate = False
        for j in kept_idx[-50:]:  # check against recent kept only for speed
            sim = float(np.dot(fp_i, fps[j]) / (norm_i * norms[j]))
            if sim > sim_threshold:
                duplicate = True
                break
        if not duplicate:
            kept_idx.append(i)
    return [patterns[i] for i in kept_idx]


# ---------------------------------------------------------------------------
# Main build loop
# ---------------------------------------------------------------------------

def build_large_library(
    output: str,
    target: int = 2000,
    steps_per_chunk: int = 8000,
    min_density: float = 0.15,
    num_workers: int = 1,
    target_chars: int = 10_000_000,
) -> int:
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    chunks = load_hf_chunks(target_chars=target_chars)
    if not chunks:
        print("[error] No data available.")
        return 1

    all_patterns: List[HierarchicalPattern] = []
    chunk_idx = 0
    t_start = time.perf_counter()

    print(f"\n[build] Target: {target} patterns | {steps_per_chunk} steps/chunk | "
          f"{num_workers} workers | min_density={min_density}")

    while len(all_patterns) < target and chunk_idx < len(chunks):
        batch_size = min(num_workers * 4, len(chunks) - chunk_idx, 32)
        batch = [
            (chunks[chunk_idx + i], steps_per_chunk, min_density, chunk_idx + i)
            for i in range(batch_size)
        ]
        chunk_idx += batch_size

        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                results = pool.map(_train_chunk, batch)
        else:
            results = [_train_chunk(b) for b in batch]

        new_patterns = [p for result in results for p in result]
        all_patterns.extend(new_patterns)
        all_patterns = deduplicate(all_patterns)

        elapsed = time.perf_counter() - t_start
        rate = len(all_patterns) / max(1, elapsed)
        eta = (target - len(all_patterns)) / max(rate, 0.001)
        print(f"[progress] chunks={chunk_idx}/{len(chunks)} "
              f"patterns={len(all_patterns)}/{target} "
              f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")

        # Save checkpoint every 500 patterns
        if len(all_patterns) >= 500 and len(all_patterns) % 500 < batch_size * 2:
            ckpt = output.replace(".pkl", f"_ckpt{len(all_patterns)}.pkl")
            PatternSerializer.save(all_patterns, ckpt)
            print(f"[checkpoint] {len(all_patterns)} patterns → {ckpt}")

    # Re-assign sequential IDs
    for i, p in enumerate(all_patterns):
        p.id = i

    PatternSerializer.save(all_patterns, output)
    elapsed = time.perf_counter() - t_start
    print(f"\n[done] {len(all_patterns)} patterns saved to {output} ({elapsed:.0f}s)")
    return 0


def _parse_args():
    p = argparse.ArgumentParser(description="Build large NLP pattern library")
    p.add_argument("--output", default="library_bootstrap/nlp_large.pkl")
    p.add_argument("--target", type=int, default=2000)
    p.add_argument("--steps-per-chunk", type=int, default=8000)
    p.add_argument("--min-density", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    p.add_argument("--target-chars", type=int, default=10_000_000)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(build_large_library(
        output=args.output,
        target=args.target,
        steps_per_chunk=args.steps_per_chunk,
        min_density=args.min_density,
        num_workers=args.workers,
        target_chars=args.target_chars,
    ))
