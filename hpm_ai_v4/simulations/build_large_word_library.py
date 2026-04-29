#!/usr/bin/env python3
"""
Build a large offline word-level HPM library from local corpora.

This mirrors the large NLP builder, but uses `WordAdapter` so the resulting
bundle is compatible with `LayeredAgent(surface_mode="word")`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.evaluators.metrics import affective_score, epistemic_score, pattern_density, social_score
from hpm_ai_v4.io.adapters import WordAdapter
from hpm_ai_v4.pattern import HierarchicalPattern
from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.tools.serializer import PatternSerializer


@dataclass
class LargeWordLibraryBuildResult:
    output: str
    pattern_count: int
    chunk_count: int
    source: str
    vocab_size: int
    registry_name: str = ""


RESERVED_TOKENS = {"<UNK>", "<BOS>", "<EOS>", "<NL>", "<PARA>"}


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_local_corpus_chunks(
    target_chars: int = 5_000_000,
    corpus_paths: Optional[Sequence[str]] = None,
) -> List[str]:
    """Load local corpus chunks without any network download path."""
    chunks: List[str] = []
    total = 0
    chunk_size = 5000

    if corpus_paths:
        for path in corpus_paths:
            if not path or not os.path.exists(path):
                continue
            text = _read_text(path)
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                if len(chunk) < 100:
                    continue
                chunks.append(chunk)
                total += len(chunk)
                if total >= target_chars:
                    break
            if total >= target_chars:
                break
        if chunks:
            return chunks

    try:
        import nltk

        corpora = [
            "brown",
            "gutenberg",
            "reuters",
            "webtext",
            "inaugural",
        ]
        for corpus_name in corpora:
            if total >= target_chars:
                break
            try:
                corpus = getattr(nltk.corpus, corpus_name)
                try:
                    text = corpus.raw()
                except Exception:
                    text = " ".join(corpus.words())
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size]
                    if len(chunk) < 100:
                        continue
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= target_chars:
                        break
            except Exception:
                continue
    except Exception:
        pass

    if not chunks:
        local = "hpm_ai_v4/simulations/data/wiki_sample.txt"
        if os.path.exists(local):
            text = _read_text(local)
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size) if len(text[i:i + chunk_size]) >= 100]

    return chunks


def _build_vocab(
    chunks: Sequence[str],
    *,
    max_vocab_size: int,
    lowercase: bool = True,
    min_freq: int = 2,
) -> Dict[str, int]:
    adapter = WordAdapter(max_vocab_size=max_vocab_size, lowercase=lowercase)
    counts: Counter[str] = Counter()
    for chunk in chunks:
        counts.update(tok for tok in adapter.tokenize(chunk) if tok not in RESERVED_TOKENS)

    vocab: Dict[str, int] = {
        "<UNK>": 0,
        "<BOS>": 1,
        "<EOS>": 2,
        "<NL>": 3,
        "<PARA>": 4,
    }
    next_idx = len(vocab)
    for token, count in counts.most_common():
        if count < min_freq:
            continue
        if token in vocab:
            continue
        if next_idx >= max_vocab_size:
            break
        vocab[token] = next_idx
        next_idx += 1
    return vocab


def _pattern_fingerprint(p: HierarchicalPattern) -> np.ndarray:
    return np.concatenate([p.A.flatten(), p.B.flatten(), p.pi.flatten()])


def deduplicate(patterns: List[HierarchicalPattern], sim_threshold: float = 0.97) -> List[HierarchicalPattern]:
    if not patterns:
        return []
    fps = [_pattern_fingerprint(p) for p in patterns]
    norms = [np.linalg.norm(fp) + 1e-12 for fp in fps]
    kept_idx: List[int] = []
    for i, (fp_i, norm_i) in enumerate(zip(fps, norms)):
        duplicate = False
        for j in kept_idx[-50:]:
            sim = float(np.dot(fp_i, fps[j]) / (norm_i * norms[j]))
            if sim > sim_threshold:
                duplicate = True
                break
        if not duplicate:
            kept_idx.append(i)
    return [patterns[i] for i in kept_idx]


def _train_chunk(args):
    chunk_text, steps, min_density, chunk_idx, keep_top_k, vocab, max_vocab_size, lowercase = args
    adapter = WordAdapter(max_vocab_size=max_vocab_size, lowercase=lowercase, vocab=vocab)
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
        d = pattern_density(p, agent.obs_buffer, [ep, aff, soc, 0.2 * soc])
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
            break
    return kept


def build_large_word_library(
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
    corpus_paths: Optional[Sequence[str]] = None,
    max_vocab_size: int = 4096,
    min_freq: int = 2,
    lowercase: bool = True,
) -> LargeWordLibraryBuildResult | int:
    base = output[:-4] if output.endswith(".pkl") else output
    os.makedirs(os.path.dirname(base) if os.path.dirname(base) else ".", exist_ok=True)

    chunks = load_local_corpus_chunks(target_chars=target_chars, corpus_paths=corpus_paths)
    if not chunks:
        print("[error] No data available.")
        return 1

    vocab = _build_vocab(chunks, max_vocab_size=max_vocab_size, lowercase=lowercase, min_freq=min_freq)
    adapter = WordAdapter(max_vocab_size=len(vocab), lowercase=lowercase, vocab=vocab)
    print(f"[data] {len(chunks)} chunks ready | vocab={len(vocab)}")

    all_patterns: List[HierarchicalPattern] = []
    chunk_idx = 0
    t_start = time.perf_counter()
    print(
        f"\n[build] Target: {target} patterns | {steps_per_chunk} steps/chunk | "
        f"{num_workers} workers | min_density={min_density} | keep_top_k={keep_top_k}"
    )

    while len(all_patterns) < target and chunk_idx < len(chunks):
        batch_size = min(num_workers * 4, len(chunks) - chunk_idx, 32)
        batch = [
            (chunks[chunk_idx + i], steps_per_chunk, min_density, chunk_idx + i, keep_top_k, vocab, len(vocab), lowercase)
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
        all_patterns = deduplicate(all_patterns, sim_threshold=dedup_threshold)

        elapsed = time.perf_counter() - t_start
        rate = len(all_patterns) / max(1, elapsed)
        eta = (target - len(all_patterns)) / max(rate, 0.001)
        print(
            f"[progress] chunks={chunk_idx}/{len(chunks)} patterns={len(all_patterns)}/{target} "
            f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
        )

        if len(all_patterns) >= 500 and len(all_patterns) % 500 < batch_size * 2:
            ckpt = base + f"_ckpt{len(all_patterns)}.l1.pkl"
            PatternSerializer.save(all_patterns, ckpt)
            print(f"[checkpoint] {len(all_patterns)} patterns → {ckpt}")

    for i, p in enumerate(all_patterns):
        p.id = i

    PatternSerializer.save(all_patterns, base + ".l1.pkl")
    surface_state = {
        "surface_mode": "word",
        "max_vocab_size": int(adapter.obs_dim),
        "lowercase": bool(lowercase),
        "word_vocab": dict(adapter._word_to_id),
    }
    with open(base + ".surface.json", "w", encoding="utf-8") as f:
        json.dump(surface_state, f)

    elapsed = time.perf_counter() - t_start
    print(f"\n[done] {len(all_patterns)} patterns saved to {base} ({elapsed:.0f}s)")
    result = LargeWordLibraryBuildResult(
        output=base,
        pattern_count=len(all_patterns),
        chunk_count=chunk_idx,
        source="local-word",
        vocab_size=len(vocab),
        registry_name="",
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
            decoder_families=["word"],
            source="local-word",
            density_mean=float(np.mean(densities)) if densities else 0.0,
            density_min=float(min(densities)) if densities else 0.0,
            density_max=float(max(densities)) if densities else 0.0,
            pattern_count=result.pattern_count,
            created_at=datetime.now(timezone.utc).isoformat(),
            notes=f"built from {result.chunk_count} chunks; word-level library",
        )
        result.registry_name = entry_name
        print(f"[registry] registered {entry_name!r} ({entry_status}) in {registry_path}")

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a large offline word-level library")
    p.add_argument("--output", default="library_bootstrap/word_large/word_large_library")
    p.add_argument("--target", type=int, default=2000)
    p.add_argument("--steps-per-chunk", type=int, default=8000)
    p.add_argument("--min-density", type=float, default=0.15)
    p.add_argument("--keep-top-k", type=int, default=4)
    p.add_argument("--dedup-threshold", type=float, default=0.97)
    p.add_argument("--promote", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    p.add_argument("--target-chars", type=int, default=10_000_000)
    p.add_argument("--registry", help="Optional JSON registry path for curated libraries")
    p.add_argument("--name", help="Registry entry name")
    p.add_argument("--domain", default="text", help="Registry domain label")
    p.add_argument("--corpus", action="append", dest="corpus_paths", help="Local corpus file; can be repeated")
    p.add_argument("--max-vocab-size", type=int, default=4096)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--no-lowercase", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = build_large_word_library(
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
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
        lowercase=not args.no_lowercase,
    )
    sys.exit(0 if isinstance(result, LargeWordLibraryBuildResult) else int(result))
