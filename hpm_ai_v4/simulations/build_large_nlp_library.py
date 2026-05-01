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
import json
import os
import sys
import time
import numpy as np
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import List, Optional
from datetime import datetime, timezone

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.io.adapters import CharClassAdapter
from hpm_ai_v4.evaluators.metrics import epistemic_score, affective_score, social_score, pattern_density
from hpm_ai_v4.tools.ingest import TextIngestGate
from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.tools.serializer import PatternSerializer
from hpm_ai_v4.pattern import HierarchicalPattern


@dataclass
class LargeNlpLibraryBuildResult:
    output: str
    pattern_count: int
    chunk_count: int
    source: str
    registry_name: str = ""


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


def _ingest_path(output: str) -> str:
    return output[:-4] + ".ingest.json" if output.endswith(".pkl") else output + ".ingest.json"


# ---------------------------------------------------------------------------
# Single-chunk training (runs in worker process)
# ---------------------------------------------------------------------------

def _train_chunk(args):
    chunk_text, steps, min_density, chunk_idx, keep_top_k = args
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
    scored = []
    for p in agent.patterns:
        if p.latent_dim <= 1:
            continue  # skip FlatPatterns
        ep = epistemic_score(p)
        aff = affective_score(p, agent.obs_buffer)
        soc = social_score(p, field_freq)
        field_infl = 0.2 * soc
        d = pattern_density(p, agent.obs_buffer, [ep, aff, soc, field_infl])
        if p.weight > 0.01:
            scored.append((float(d), p))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    kept: List[HierarchicalPattern] = []
    for density, pattern in scored:
        if density >= min_density or len(kept) < keep_top_k:
            pattern.source_corpus = f"chunk_{chunk_idx}"
            pattern.density_at_save = float(density)
            kept.append(pattern)
        if len(kept) >= keep_top_k and density < min_density:
            # Once we have the floor and are past the density gate, stop early.
            break

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
    keep_top_k: int = 4,
    dedup_threshold: float = 0.97,
    promote: bool = False,
    num_workers: int = 1,
    target_chars: int = 10_000_000,
    registry_path: Optional[str] = None,
    name: Optional[str] = None,
    domain: str = "text",
) -> LargeNlpLibraryBuildResult | int:
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    chunks = load_hf_chunks(target_chars=target_chars)
    if not chunks:
        print("[error] No data available.")
        return 1

    ingest_gate = TextIngestGate.load_snapshot_from_path(_ingest_path(output), adapter=CharClassAdapter())
    all_patterns: List[HierarchicalPattern] = []
    chunk_idx = 0
    t_start = time.perf_counter()

    print(f"\n[build] Target: {target} patterns | {steps_per_chunk} steps/chunk | "
          f"{num_workers} workers | min_density={min_density} | keep_top_k={keep_top_k}")

    while len(all_patterns) < target and chunk_idx < len(chunks):
        batch_size = min(num_workers * 4, len(chunks) - chunk_idx, 32)
        batch = []
        while chunk_idx < len(chunks) and len(batch) < batch_size:
            chunk_text = chunks[chunk_idx]
            current_idx = chunk_idx
            chunk_idx += 1
            if not ingest_gate.register_text(chunk_text):
                continue
            batch.append((chunk_text, steps_per_chunk, min_density, current_idx, keep_top_k))
        if not batch:
            continue

        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                results = pool.map(_train_chunk, batch)
        else:
            results = [_train_chunk(b) for b in batch]

        new_patterns = [p for result in results for p in result]
        new_patterns = ingest_gate.filter_new_patterns(new_patterns)
        all_patterns.extend(new_patterns)
        all_patterns = deduplicate(all_patterns, sim_threshold=dedup_threshold)

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
    result = LargeNlpLibraryBuildResult(
        output=output,
        pattern_count=len(all_patterns),
        chunk_count=chunk_idx,
        source="nltk",
        registry_name="",
    )

    if registry_path:
        densities = [float(getattr(p, "density_at_save", 0.0)) for p in all_patterns]
        registry = LibraryRegistry(registry_path)
        entry_name = name or os.path.splitext(os.path.basename(output))[0]
        entry_status = "promoted" if promote or result.pattern_count >= 2000 else "seed"
        registry.upsert(
            name=entry_name,
            path=output,
            domain=domain,
            status=entry_status,
            bundle_kind="flat",
            level_contract="l1",
            obs_dims=[5],
            source="nltk",
            density_mean=float(np.mean(densities)) if densities else 0.0,
            density_min=float(min(densities)) if densities else 0.0,
            density_max=float(max(densities)) if densities else 0.0,
            pattern_count=result.pattern_count,
            created_at=datetime.now(timezone.utc).isoformat(),
            notes=f"built from {result.chunk_count} chunks; base text library",
            ingest_state=ingest_gate.snapshot(),
        )
        result.registry_name = entry_name
        print(f"[registry] registered {entry_name!r} ({entry_status}) in {registry_path}")

    with open(_ingest_path(output), "w", encoding="utf-8") as f:
        json.dump(ingest_gate.snapshot(), f, sort_keys=True)

    return result


def _parse_args():
    p = argparse.ArgumentParser(description="Build large NLP pattern library")
    p.add_argument("--output", default="library_bootstrap/nlp_large.pkl")
    p.add_argument("--target", type=int, default=2000)
    p.add_argument("--steps-per-chunk", type=int, default=8000)
    p.add_argument("--min-density", type=float, default=0.15)
    p.add_argument("--keep-top-k", type=int, default=4)
    p.add_argument("--dedup-threshold", type=float, default=0.97)
    p.add_argument("--promote", action="store_true", help="Register the built library as promoted")
    p.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    p.add_argument("--target-chars", type=int, default=10_000_000)
    p.add_argument("--registry", help="Optional JSON registry path for curated libraries")
    p.add_argument("--name", help="Registry entry name")
    p.add_argument("--domain", default="text", help="Registry domain label")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = build_large_library(
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
    )
    sys.exit(0 if isinstance(result, LargeNlpLibraryBuildResult) else int(result))
