#!/usr/bin/env python3
"""
Quality report for a saved HPM pattern library.

Usage:
    python3 -m hpm_ai_v4.tools.library_quality <path.pkl> [--top 10]
"""
import argparse
import sys
import numpy as np
from typing import List

from hpm_ai_v4.tools.serializer import PatternSerializer
from hpm_ai_v4.pattern import HierarchicalPattern
from hpm_ai_v4.evaluators.metrics import (
    epistemic_score, hierarchical_compression_score, compression_gate,
)


def _fp(p: HierarchicalPattern) -> np.ndarray:
    return np.concatenate([p.A.flatten(), p.B.flatten(), p.pi.flatten()])


def diversity_score(patterns: List[HierarchicalPattern], sample: int = 200) -> float:
    """Mean pairwise cosine distance on a random sample (1 = fully diverse, 0 = identical)."""
    if len(patterns) < 2:
        return 0.0
    rng = np.random.default_rng(42)
    idx = rng.choice(len(patterns), size=min(sample, len(patterns)), replace=False)
    fps = np.array([_fp(patterns[i]) for i in idx], dtype=np.float32)
    norms = np.linalg.norm(fps, axis=1, keepdims=True) + 1e-12
    fps_n = fps / norms
    sim_mat = fps_n @ fps_n.T
    n = len(fps_n)
    upper = sim_mat[np.triu_indices(n, k=1)]
    return float(1.0 - upper.mean())


def report(path: str, top_k: int = 10) -> None:
    patterns: List[HierarchicalPattern] = PatternSerializer.load(path)
    if not patterns:
        print(f"[error] No patterns loaded from {path}")
        return

    hier = [p for p in patterns if p.latent_dim > 1]
    flat = [p for p in patterns if p.latent_dim <= 1]

    weights = np.array([p.weight for p in patterns])
    k_vals = [p.latent_dim for p in hier]
    losses = [p.running_loss for p in hier]
    compressions = [p.compression() for p in hier]
    ep_scores = [epistemic_score(p) for p in hier]
    comp_scores = [hierarchical_compression_score(p) for p in hier]
    gates = [compression_gate(p) for p in hier]
    densities = [getattr(p, "density_at_save", None) for p in hier]
    densities = [d for d in densities if d is not None]

    div = diversity_score(hier)

    print(f"\n{'='*60}")
    print(f"  Pattern Library Quality Report")
    print(f"  File: {path}")
    print(f"{'='*60}")
    print(f"\n  Total patterns    : {len(patterns)}")
    print(f"  HierarchicalPatterns (K>1): {len(hier)}")
    print(f"  FlatPatterns (K=1)        : {len(flat)}")

    if hier:
        print(f"\n--- Structure ---")
        from collections import Counter
        k_dist = Counter(k_vals)
        for k in sorted(k_dist):
            print(f"  K={k}: {k_dist[k]} patterns ({100*k_dist[k]/len(hier):.1f}%)")

        print(f"\n--- Weights ---")
        print(f"  Mean   : {weights.mean():.4f}")
        print(f"  Std    : {weights.std():.4f}")
        print(f"  Max    : {weights.max():.4f}  Min: {weights.min():.4f}")
        top_weight_idx = np.argsort(weights)[-3:][::-1]
        print(f"  Top-3 weight patterns: {[patterns[i].id for i in top_weight_idx]}")

        print(f"\n--- Epistemic (running loss, lower=better) ---")
        print(f"  Mean loss : {np.mean(losses):.4f}")
        print(f"  Std loss  : {np.std(losses):.4f}")
        print(f"  % loss<0.3: {100*np.mean(np.array(losses)<0.3):.1f}%")
        print(f"  % loss<0.5: {100*np.mean(np.array(losses)<0.5):.1f}%")

        print(f"\n--- Compression ---")
        print(f"  Mean compression   : {np.mean(compressions):.4f}")
        print(f"  Mean comp_gate     : {np.mean(gates):.4f}  (1=ready, 0=not epistemically valid)")
        print(f"  Mean hier_comp_score: {np.mean(comp_scores):.4f}")
        print(f"  % compression>0.1  : {100*np.mean(np.array(compressions)>0.1):.1f}%")

        print(f"\n--- Diversity ---")
        print(f"  Pairwise cosine distance (sample=200): {div:.4f}  (0=identical, 1=fully diverse)")

        if densities:
            print(f"\n--- Density at save ---")
            print(f"  Mean: {np.mean(densities):.4f}  Std: {np.std(densities):.4f}")
            print(f"  Min : {np.min(densities):.4f}  Max: {np.max(densities):.4f}")

        print(f"\n--- Top {top_k} patterns by hier_comp_score ---")
        top_idx = np.argsort(comp_scores)[-top_k:][::-1]
        print(f"  {'id':>5}  {'K':>2}  {'loss':>6}  {'comp':>6}  {'gate':>5}  {'hcs':>6}  {'weight':>7}")
        for i in top_idx:
            p = hier[i]
            print(f"  {p.id:>5}  {p.latent_dim:>2}  {losses[i]:>6.3f}  "
                  f"{compressions[i]:>6.3f}  {gates[i]:>5.3f}  {comp_scores[i]:>6.3f}  {p.weight:>7.4f}")

    print(f"\n{'='*60}\n")


def main():
    p = argparse.ArgumentParser(description="HPM pattern library quality report")
    p.add_argument("path", help="Path to .pkl pattern library")
    p.add_argument("--top", type=int, default=10, help="Top N patterns to show")
    args = p.parse_args()
    report(args.path, top_k=args.top)


if __name__ == "__main__":
    main()
