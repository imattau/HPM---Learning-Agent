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
import re
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
from hpm_ai_v4.tools.ingest import TextIngestGate
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


def _tokenize_raw_words(text: str, lowercase: bool = True) -> List[str]:
    raw = str(text or "")
    tokens: List[str] = []
    for tok in WordAdapter.TOKEN_RE.findall(raw):
        if not tok or tok.isspace():
            continue
        if tok == "\n":
            continue
        tokens.append(tok.lower() if lowercase else tok)
    return tokens


def _weighted_jaccard(a: Counter[str], b: Counter[str], limit: int = 12) -> float:
    if not a or not b:
        return 0.0
    top_a = dict(a.most_common(limit))
    top_b = dict(b.most_common(limit))
    keys = set(top_a) | set(top_b)
    if not keys:
        return 0.0
    intersection = sum(min(top_a.get(key, 0), top_b.get(key, 0)) for key in keys)
    union = sum(max(top_a.get(key, 0), top_b.get(key, 0)) for key in keys)
    return float(intersection / max(1, union))


def _ingest_path(base: str) -> str:
    return base[:-4] + ".ingest.json" if base.endswith(".pkl") else base + ".ingest.json"


def _content_neighbors(counter: Counter[str]) -> Counter[str]:
    filtered: Counter[str] = Counter()
    for token, count in counter.items():
        if not token or len(token) < 3:
            continue
        bucket = WordAdapter.semantic_bucket(token)
        if bucket in {0, 1, 2, 3}:
            continue
        filtered[token] = count
    return filtered


def _sentence_chunks(text: str) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = re.split(r"(?<=[.!?])\s+", raw)
    return [part.strip() for part in parts if part and part.strip()]


def _stem_like(token: str) -> str:
    tok = str(token or "").strip().lower()
    if len(tok) <= 3:
        return tok
    if tok.endswith("'s"):
        tok = tok[:-2]
    for suffix in ("ing", "ed", "es", "s"):
        if tok.endswith(suffix) and len(tok) > len(suffix) + 2:
            base = tok[: -len(suffix)]
            if suffix == "es" and base.endswith("i"):
                base = base[:-1] + "y"
            return base
    return tok


def _build_corpus_aliases(
    chunks: Sequence[str],
    *,
    lowercase: bool = True,
    min_freq: int = 2,
    max_aliases: int = 512,
    similarity_threshold: float = 0.36,
) -> Dict[str, str]:
    counts: Counter[str] = Counter()
    contexts: Dict[str, Counter[str]] = {}
    sentence_contexts: Dict[str, Counter[str]] = {}

    for chunk in chunks:
        tokens = _tokenize_raw_words(chunk, lowercase=lowercase)
        if len(tokens) < 2:
            continue
        counts.update(tokens)
        for idx, tok in enumerate(tokens):
            ctx = contexts.setdefault(tok, Counter())
            left = tokens[max(0, idx - 2): idx]
            right = tokens[idx + 1: idx + 3]
            for neighbor in left + right:
                if neighbor != tok:
                    ctx[neighbor] += 1
        for sentence in _sentence_chunks(chunk):
            sentence_tokens = _tokenize_raw_words(sentence, lowercase=lowercase)
            if len(sentence_tokens) < 2:
                continue
            sentence_content = _content_neighbors(Counter(sentence_tokens))
            for tok in sentence_tokens:
                sent_ctx = sentence_contexts.setdefault(tok, Counter())
                for neighbor, value in sentence_content.items():
                    if neighbor != tok:
                        sent_ctx[neighbor] += value

    ordered = sorted(
        [tok for tok, count in counts.items() if count >= min_freq],
        key=lambda tok: (-counts[tok], WordAdapter.semantic_bucket(tok), tok),
    )

    alias_map: Dict[str, str] = {}
    canonical_heads = set()
    alias_terms = set()

    def _canonical_choice(a: str, b: str) -> str:
        return min(
            (a, b),
            key=lambda tok: (-counts[tok], len(tok), WordAdapter.semantic_bucket(tok), tok),
        )

    candidate_pool = [
        tok
        for tok in ordered
        if len(tok) >= 3
        and WordAdapter.semantic_bucket(tok) not in {0, 1, 2, 3}
        and tok not in WordAdapter.COMMON_VERBS
        and tok not in WordAdapter.COMMON_ADJECTIVES
    ]

    candidate_scores: Dict[str, List[Tuple[float, str, int]]] = {}
    token_context_cache: Dict[str, Counter[str]] = {
        tok: _content_neighbors(contexts.get(tok, Counter()))
        for tok in candidate_pool
    }
    token_sentence_context_cache: Dict[str, Counter[str]] = {
        tok: _content_neighbors(sentence_contexts.get(tok, Counter()))
        for tok in candidate_pool
    }

    for token in sorted(candidate_pool, key=lambda tok: (counts[tok], WordAdapter.semantic_bucket(tok), tok)):
        if token in alias_map:
            continue
        bucket = WordAdapter.semantic_bucket(token)
        stem = _stem_like(token)
        token_ctx = token_context_cache.get(token, Counter())
        token_sent_ctx = token_sentence_context_cache.get(token, Counter())
        scored_candidates: List[Tuple[float, str, int]] = []
        for candidate in ordered:
            if candidate == token or counts[candidate] < counts[token]:
                continue
            if WordAdapter.semantic_bucket(candidate) != bucket:
                continue
            if candidate in WordAdapter.COMMON_VERBS or candidate in WordAdapter.COMMON_ADJECTIVES:
                continue
            candidate_stem = _stem_like(candidate)
            candidate_ctx = token_context_cache.get(candidate, Counter())
            candidate_sent_ctx = token_sentence_context_cache.get(candidate, Counter())
            shared_neighbors = token_ctx.keys() & candidate_ctx.keys()
            shared_sent_neighbors = token_sent_ctx.keys() & candidate_sent_ctx.keys()
            overlap = len(shared_neighbors) + len(shared_sent_neighbors)
            if overlap < 1:
                continue
            score = 0.65 * _weighted_jaccard(token_ctx, candidate_ctx)
            if token_sent_ctx and candidate_sent_ctx:
                score += 0.35 * _weighted_jaccard(token_sent_ctx, candidate_sent_ctx)
            if stem == candidate_stem:
                score += 0.18
            if stem and candidate_stem and stem[:4] == candidate_stem[:4]:
                score += 0.08
            if abs(len(candidate) - len(token)) <= 3:
                score += 0.04
            scored_candidates.append((score, candidate, overlap))
        if not scored_candidates:
            continue
        scored_candidates.sort(key=lambda item: (item[0], counts[item[1]], item[1]), reverse=True)
        candidate_scores[token] = scored_candidates

    def _best_choice(token: str) -> Tuple[str, float, int, float]:
        scored = candidate_scores.get(token, [])
        if not scored:
            return "", 0.0, 0, 0.0
        score, candidate, overlap = scored[0]
        second_best = scored[1][0] if len(scored) > 1 else 0.0
        return candidate, score, overlap, second_best

    for token in sorted(candidate_scores, key=lambda tok: (counts[tok], WordAdapter.semantic_bucket(tok), tok)):
        if token in alias_terms or token in canonical_heads:
            continue
        best_candidate, best_score, best_overlap, second_best = _best_choice(token)
        if not best_candidate:
            continue
        if best_score < similarity_threshold:
            continue
        if best_score < second_best + 0.12:
            continue
        if best_overlap < 1:
            continue
        canonical = _canonical_choice(token, best_candidate)
        alias = token if canonical == best_candidate else best_candidate
        if alias == canonical:
            continue
        if canonical in alias_terms:
            continue
        alias_map[alias] = canonical
        alias_terms.add(alias)
        canonical_heads.add(canonical)
        if len(alias_map) >= max_aliases:
            break

    return alias_map


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

    corpus_aliases = _build_corpus_aliases(
        chunks,
        lowercase=lowercase,
        min_freq=min_freq,
        max_aliases=max(64, max_vocab_size // 8),
    )
    merged_aliases = dict(WordAdapter.DEFAULT_CANONICAL_ALIASES)
    merged_aliases.update(corpus_aliases)
    vocab = WordAdapter.build_semantic_vocab(
        chunks,
        max_vocab_size=max_vocab_size,
        lowercase=lowercase,
        min_freq=min_freq,
        canonical_aliases=merged_aliases,
    )
    adapter = WordAdapter(
        max_vocab_size=max_vocab_size,
        lowercase=lowercase,
        vocab=vocab,
        canonical_aliases=merged_aliases,
    )
    ingest_gate = TextIngestGate.load_snapshot_from_path(_ingest_path(base), adapter=adapter, lowercase=lowercase)
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
        batch = []
        while chunk_idx < len(chunks) and len(batch) < batch_size:
            chunk_text = chunks[chunk_idx]
            current_idx = chunk_idx
            chunk_idx += 1
            if not ingest_gate.register_text(chunk_text):
                continue
            batch.append(
                (chunk_text, steps_per_chunk, min_density, current_idx, keep_top_k, vocab, len(vocab), lowercase)
            )
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
        "canonical_aliases": dict(adapter._canonical_aliases),
        "derived_corpus_aliases": dict(corpus_aliases),
        "vocab_contract": WordAdapter.vocab_contract(),
        "ingest": ingest_gate.snapshot(),
    }
    with open(base + ".surface.json", "w", encoding="utf-8") as f:
        json.dump(surface_state, f)
    with open(_ingest_path(base), "w", encoding="utf-8") as f:
        json.dump(ingest_gate.snapshot(), f, sort_keys=True)

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
            ingest_state=ingest_gate.snapshot(),
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
