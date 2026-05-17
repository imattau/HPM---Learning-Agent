#!/usr/bin/env python3
"""Inspect the local HPM pattern and reasoning graph libraries.

This script prints:
- the shared cross-agent PatternStore contents
- each agent's paged SQLite archive
- the reasoning graph snapshot and graph summary

Usage:
    python scripts/inspect_pattern_graph.py
    python scripts/inspect_pattern_graph.py --corpus /path/to/corpus.txt
    python scripts/inspect_pattern_graph.py --cache-dir /path/to/.hpm_pattern_cache
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _default_corpus_path() -> str:
    base = Path(__file__).resolve().parents[1] / "hpm_ai_v6" / "data" / "corpus" / "alice_mini.txt"
    return str(base)


def _default_cache_dir(corpus_path: str) -> str:
    corpus_dir = Path(corpus_path).resolve().parent
    return str(corpus_dir / ".hpm_pattern_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect HPM pattern and reasoning graph storage.")
    parser.add_argument("--corpus", default=_default_corpus_path(), help="Corpus path used to initialise MultiAgentReader.")
    parser.add_argument("--cache-dir", default=None, help="Pattern cache directory. Defaults to the reader's cache path.")
    parser.add_argument("--top-k", type=int, default=10, help="How many entries to print per store.")
    parser.add_argument("--no-warm-start", action="store_true", help="Skip hydrating the reader from cache.")
    return parser.parse_args()


def _format_vector(vec: np.ndarray, max_items: int = 4) -> str:
    arr = np.asarray(vec, dtype=float).flatten()
    head = ", ".join(f"{value:.3f}" for value in arr[:max_items])
    suffix = ", ..." if arr.size > max_items else ""
    return f"[{head}{suffix}]"


def _print_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _print_shared_store(reader, top_k: int) -> None:
    from hpm_ai_v6.hpm_model.storage.pattern_store import PatternStore

    store = reader.pattern_store
    patterns, weights = store.load()
    _print_header("Shared PatternStore")
    print(f"path: {store._store_path}")  # local inspection tool
    print(f"entries: {len(patterns)}")
    if not patterns:
        return

    ranked = sorted(zip(patterns, weights), key=lambda item: float(item[1]), reverse=True)[:top_k]
    for idx, (cell, weight) in enumerate(ranked, start=1):
        print(f"{idx:>2}. {cell.name}  weight={float(weight):.3f}  emb={_format_vector(cell.as_numpy())}")


def _print_agent_pagers(reader, top_k: int) -> None:
    _print_header("Agent Pagers")
    for name, agent in reader.agents.items():
        pager = getattr(agent, "pattern_pager", None)
        if pager is None:
            continue
        try:
            payloads = pager.iter_index_payloads()
        except Exception as exc:
            print(f"{name}: error reading pager index: {exc}")
            continue

        print(f"{name}: {len(payloads)} entries")
        for idx, payload in enumerate(sorted(payloads, key=lambda p: float(p.get("weight", 0.0)), reverse=True)[:top_k], start=1):
            print(
                f"  {idx:>2}. {payload.get('name', '')}  "
                f"weight={float(payload.get('weight', 0.0)):.3f}  "
                f"dim={payload.get('dim', '?')}"
            )


def _print_reasoning_graph(reader, top_k: int) -> None:
    reasoning_agent = getattr(reader, "reasoning_agent", None)
    if reasoning_agent is None:
        return

    _print_header("Reasoning Graph")
    if hasattr(reasoning_agent, "index_status"):
        try:
            print(f"status: {reasoning_agent.index_status()}")
        except Exception as exc:
            print(f"status: error: {exc}")

    if hasattr(reasoning_agent, "analyse_patterns"):
        try:
            report = reasoning_agent.analyse_patterns()
        except Exception as exc:
            print(f"analyse_patterns: error: {exc}")
            return

        print(f"total_patterns: {report.get('total_patterns', 0)}")
        print(f"total_nodes: {report.get('total_nodes', 0)}")
        print(f"total_edges: {report.get('total_edges', 0)}")
        agents = report.get("agents", {}) or {}
        if agents:
            print("agents:")
            for agent_name, stats in agents.items():
                print(f"  {agent_name}: {stats}")
        relations = report.get("relations", {}) or {}
        if relations:
            print("relations:")
            for relation, count in sorted(relations.items(), key=lambda item: item[1], reverse=True)[:top_k]:
                print(f"  {relation}: {count}")
        hubs = report.get("top_hubs", []) or []
        if hubs:
            print("top_hubs:")
            for hub in hubs[:top_k]:
                print(f"  - {hub}")
        components = report.get("components", {}) or {}
        if components:
            print(f"components: {components}")


def _print_reasoning_graph_store(reader) -> None:
    reasoning_agent = getattr(reader, "reasoning_agent", None)
    if reasoning_agent is None:
        return
    store = getattr(reasoning_agent, "_graph_store", None)
    if store is None:
        return
    _print_header("ReasoningGraphStore")
    print(f"path: {store.db_path}")
    snapshot = store.load()
    if snapshot is None:
        print("snapshot: none")
        return
    print(f"signature: {snapshot.signature}")
    print(f"updated_at: {snapshot.updated_at}")


def main() -> None:
    args = parse_args()
    corpus_path = os.path.abspath(args.corpus)
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    cache_dir = os.path.abspath(args.cache_dir) if args.cache_dir else None

    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader

    reader = MultiAgentReader(
        corpus_path,
        pattern_cache_dir=cache_dir,
        warm_start=not args.no_warm_start,
    )

    print(f"corpus: {corpus_path}")
    print(f"cache_dir: {reader.pattern_cache_dir}")

    _print_shared_store(reader, args.top_k)
    _print_agent_pagers(reader, args.top_k)
    _print_reasoning_graph_store(reader)
    _print_reasoning_graph(reader, args.top_k)


if __name__ == "__main__":
    main()
