#!/usr/bin/env python3
"""
Build a learned substrate library by mining stable character n-gram merges.

This is the substrate-level analogue of the word builder:
- probe an ASCII surface agent to collect latent-state alignments
- fit conservative merge rules from aligned character spans
- train a fresh library on the merged token stream
- persist the merge table alongside the L1 bundle
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.evaluators.metrics import affective_score, epistemic_score, pattern_density, social_score
from hpm_ai_v4.io.adapters import AsciiCharAdapter, LearnedSubstrateAdapter, SubstrateMergeRule
from hpm_ai_v4.pattern import HierarchicalPattern
from hpm_ai_v4.simulations.build_large_word_library import deduplicate, load_local_corpus_chunks
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.ingest import TextIngestGate
from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.tools.serializer import PatternSerializer


@dataclass(frozen=True)
class LearnedSubstrateLibraryBuildResult:
    output: str
    pattern_count: int
    chunk_count: int
    source: str
    merge_count: int
    registry_name: str = ""


def _project_ascii(text: str) -> str:
    return "".join(ch for ch in str(text or "") if ch == "\n" or 32 <= ord(ch) <= 126)


def _probe_latent_paths(chunks: Sequence[str]) -> Tuple[List[str], List[List[int]]]:
    probe = LayeredAgent(num_workers=1, surface_mode="ascii")
    texts: List[str] = []
    paths: List[List[int]] = []
    for chunk in chunks:
        projected = _project_ascii(chunk)
        if len(projected) < 32:
            continue
        start = len(probe._l1_state_history)
        probe.observe_text(projected, feedback_mode="target")
        path = probe._l1_state_history[start:]
        if len(path) >= 16:
            texts.append(projected)
            paths.append([int(v) for v in path[:len(projected)]])
    return texts, paths


def _train_chunk(chunk_text: str, steps: int, adapter: LearnedSubstrateAdapter, chunk_idx: int, min_density: float, keep_top_k: int) -> List[HierarchicalPattern]:
    tokens = adapter.to_observations(chunk_text, max_length=max(200, steps))
    if len(tokens) < 20:
        return []

    agent = HPMAgent(num_initial_patterns=20, obs_dim=adapter.obs_dim)
    n = len(tokens)
    for step in range(steps):
        agent.perceive_and_learn(int(tokens[step % n]))

    field_freq = {p.id: p.weight for p in agent.patterns}
    scored = []
    for p in agent.patterns:
        ep = epistemic_score(p)
        aff = affective_score(p, agent.obs_buffer)
        soc = social_score(p, field_freq)
        density = pattern_density(p, agent.obs_buffer, [ep, aff, soc, 0.2 * soc])
        scored.append((float(density), p))

    scored.sort(key=lambda item: item[0], reverse=True)
    kept: List[HierarchicalPattern] = []
    for density, pattern in scored:
        if density >= min_density or len(kept) < keep_top_k:
            pattern.source_corpus = f"chunk_{chunk_idx}"
            pattern.density_at_save = float(density)
            kept.append(pattern)
        if len(kept) >= keep_top_k and density < min_density:
            break
    return kept


def _fallback_merge_rules(texts: Sequence[str], max_merges: int) -> Dict[str, Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for text in texts:
        projected = _project_ascii(text).lower()
        for token in re.findall(r"[A-Za-z]{3,8}", projected):
            counts[token] = counts.get(token, 0) + 1
    if not counts:
        return {}
    ordered = sorted(counts.items(), key=lambda item: (-item[1], len(item[0]), item[0]))
    merges: Dict[str, Dict[str, Any]] = {}
    for token, support in ordered:
        if support < 2:
            continue
        merges[token] = {
            "latent_state": 0,
            "support": int(support),
            "purity": 1.0,
            "span": len(token),
        }
        if len(merges) >= max_merges:
            break
    return merges


def build_learned_substrate_library(
    output: str,
    target: int = 2000,
    steps_per_chunk: int = 6000,
    min_density: float = 0.15,
    keep_top_k: int = 4,
    dedup_threshold: float = 0.97,
    promote: bool = False,
    num_workers: int = 1,
    target_chars: int = 5_000_000,
    registry_path: Optional[str] = None,
    name: Optional[str] = None,
    domain: str = "text",
    corpus_paths: Optional[Sequence[str]] = None,
    max_merges: int = 64,
    min_support: int = 5,
    purity_threshold: float = 0.95,
) -> LearnedSubstrateLibraryBuildResult | int:
    base = output[:-4] if output.endswith(".pkl") else output
    os.makedirs(os.path.dirname(base) if os.path.dirname(base) else ".", exist_ok=True)

    chunks = load_local_corpus_chunks(target_chars=target_chars, corpus_paths=corpus_paths)
    if not chunks:
        print("[error] No data available.")
        return 1

    probe_texts, probe_paths = _probe_latent_paths(chunks[: max(8, min(len(chunks), 64))])
    adapter = LearnedSubstrateAdapter(max_merges=max_merges, lowercase=True)
    merge_count = adapter.fit_merges(
        probe_texts,
        probe_paths,
        min_support=min_support,
        purity_threshold=purity_threshold,
        max_ngram=4,
    )
    if merge_count == 0:
        fallback_rules = _fallback_merge_rules(probe_texts or chunks, max_merges=max_merges)
        if fallback_rules:
            adapter._merge_rules = {
                token: SubstrateMergeRule(
                    token=token,
                    latent_state=int(rule["latent_state"]),
                    support=int(rule["support"]),
                    purity=float(rule["purity"]),
                    span=int(rule["span"]),
                )
                for token, rule in fallback_rules.items()
            }
            adapter._refresh_merge_vocab()
            merge_count = len(adapter._merge_rules)

    print(f"[data] {len(chunks)} chunks ready | merges={merge_count} | obs_dim={adapter.obs_dim}")

    ingest_gate = TextIngestGate.load_snapshot_from_path(base + ".ingest.json", adapter=adapter, lowercase=True)
    all_patterns: List[HierarchicalPattern] = []
    chunk_idx = 0
    t_start = time.perf_counter()
    while len(all_patterns) < target and chunk_idx < len(chunks):
        chunk = _project_ascii(chunks[chunk_idx])
        chunk_idx += 1
        if not ingest_gate.register_text(chunk):
            continue
        new_patterns = _train_chunk(chunk, steps_per_chunk, adapter, chunk_idx, min_density, keep_top_k)
        new_patterns = ingest_gate.filter_new_patterns(new_patterns)
        all_patterns.extend(new_patterns)
        all_patterns = deduplicate(all_patterns, sim_threshold=dedup_threshold)
        elapsed = time.perf_counter() - t_start
        rate = len(all_patterns) / max(1, elapsed)
        eta = (target - len(all_patterns)) / max(rate, 0.001)
        print(
            f"[progress] chunks={chunk_idx}/{len(chunks)} patterns={len(all_patterns)}/{target} "
            f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
        )

    for i, p in enumerate(all_patterns):
        p.id = i

    PatternSerializer.save(all_patterns, base + ".l1.pkl")
    surface_state = {
        "surface_mode": "merged",
        "merge_max_merges": int(max_merges),
        "merge_rules": adapter.merge_rules(),
        "learned_surface_contract": {
            "stable_tokenization": True,
            "bounded_vocab": True,
            "token_merge_mode": "char_ngram_to_surface_tokens",
            "bundle_keys": ["merge_max_merges", "merge_rules", "surface_mode"],
        },
        "ingest": ingest_gate.snapshot(),
    }
    with open(base + ".surface.json", "w", encoding="utf-8") as f:
        json.dump(surface_state, f)
    with open(base + ".ingest.json", "w", encoding="utf-8") as f:
        json.dump(ingest_gate.snapshot(), f, sort_keys=True)

    elapsed = time.perf_counter() - t_start
    print(f"\n[done] {len(all_patterns)} patterns saved to {base} ({elapsed:.0f}s)")
    result = LearnedSubstrateLibraryBuildResult(
        output=base,
        pattern_count=len(all_patterns),
        chunk_count=chunk_idx,
        source="local-merged-substrate",
        merge_count=merge_count,
    )

    if registry_path:
        densities = [float(getattr(p, "density_at_save", 0.0)) for p in all_patterns]
        registry = LibraryRegistry(registry_path)
        entry_name = name or os.path.splitext(os.path.basename(base))[0]
        entry_status = "promoted" if promote or result.pattern_count >= 2000 else "seed"
        registry.upsert(
            name=entry_name,
            path=base,
            domain=domain,
            status=entry_status,
            bundle_kind="stacked",
            level_contract="l1",
            obs_dims=[adapter.obs_dim],
            decoder_families=["char", "word"],
            source="local-merged-substrate",
            density_mean=float(np.mean(densities)) if densities else 0.0,
            density_min=float(min(densities)) if densities else 0.0,
            density_max=float(max(densities)) if densities else 0.0,
            pattern_count=result.pattern_count,
            created_at=datetime.now(timezone.utc).isoformat(),
            notes=f"built from {result.chunk_count} chunks; merged substrate with {merge_count} merge rules",
            ingest_state=ingest_gate.snapshot(),
        )
        result = LearnedSubstrateLibraryBuildResult(
            output=result.output,
            pattern_count=result.pattern_count,
            chunk_count=result.chunk_count,
            source=result.source,
            merge_count=result.merge_count,
            registry_name=entry_name,
        )
        print(f"[registry] registered {entry_name!r} ({entry_status}) in {registry_path}")

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a learned merged-substrate library")
    p.add_argument("--output", default="library_bootstrap/merged_large/merged_large_library")
    p.add_argument("--target", type=int, default=2000)
    p.add_argument("--steps-per-chunk", type=int, default=6000)
    p.add_argument("--min-density", type=float, default=0.15)
    p.add_argument("--keep-top-k", type=int, default=4)
    p.add_argument("--dedup-threshold", type=float, default=0.97)
    p.add_argument("--promote", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--target-chars", type=int, default=5_000_000)
    p.add_argument("--registry", help="Optional JSON registry path for curated libraries")
    p.add_argument("--name", help="Registry entry name")
    p.add_argument("--domain", default="text", help="Registry domain label")
    p.add_argument("--corpus", action="append", dest="corpus_paths", help="Local corpus file; can be repeated")
    p.add_argument("--max-merges", type=int, default=64)
    p.add_argument("--min-support", type=int, default=5)
    p.add_argument("--purity-threshold", type=float, default=0.95)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = build_learned_substrate_library(
        output=args.output,
        target=args.target,
        steps_per_chunk=args.steps_per_chunk,
        min_density=args.min_density,
        keep_top_k=args.keep_top_k,
        dedup_threshold=args.dedup_threshold,
        promote=args.promote,
        num_workers=args.workers,
        target_chars=args.target_chars,
        registry_path=args.registry,
        name=args.name,
        domain=args.domain,
        corpus_paths=args.corpus_paths,
        max_merges=args.max_merges,
        min_support=args.min_support,
        purity_threshold=args.purity_threshold,
    )
    sys.exit(0 if isinstance(result, LearnedSubstrateLibraryBuildResult) else int(result))
