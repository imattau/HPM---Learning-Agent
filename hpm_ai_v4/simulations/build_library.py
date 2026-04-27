#!/usr/bin/env python3
"""
Offline pattern library creation script.

Usage:
    PYTHONPATH=. python3 hpm_ai_v4/simulations/build_library.py \
        --corpus wiki_sample.txt \
        --output wiki_patterns.pkl \
        --steps 100000 \
        --min-density 0.3
"""
import argparse
import os
import sys
import json
import urllib.request
import urllib.parse
import numpy as np
from datetime import datetime, timezone
from typing import Optional

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.io.adapters import CharClassAdapter
from hpm_ai_v4.evaluators.metrics import pattern_density, affective_score, social_score
from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.tools.serializer import PatternSerializer


SYNTHETIC_CORPUS = (
    "the cat sat on the mat. the cat ate the rat. "
    "a bat sat on a flat mat. the fat cat and the rat. "
) * 200


def fetch_wikipedia_content(topic: str) -> str:
    """Search for a topic on Wikipedia and fetch the full text of the first match."""
    print(f"[wiki] Searching for topic: {topic!r}...")
    
    headers = {
        'User-Agent': 'HPM-Learning-Agent/1.0 (imatt.au@protonmail.com)'
    }
    
    # 1. Search for the best matching page title
    search_url = (
        "https://en.wikipedia.org/w/api.php?action=query&list=search"
        f"&srsearch={urllib.parse.quote(topic)}&format=json"
    )
    
    try:
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req) as response:
            search_data = json.loads(response.read().decode())
            search_results = search_data.get('query', {}).get('search', [])
            
        if not search_results:
            print(f"[error] No Wikipedia pages found for topic: {topic}")
            return ""
            
        page_title = search_results[0]['title']
        print(f"[wiki] Found page: {page_title!r}. Fetching content...")
        
        # 2. Fetch the text content of that page
        content_url = (
            "https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
            f"&explaintext=1&titles={urllib.parse.quote(page_title)}&format=json"
        )
        
        req = urllib.request.Request(content_url, headers=headers)
        with urllib.request.urlopen(req) as response:
            content_data = json.loads(response.read().decode())
            pages = content_data.get('query', {}).get('pages', {})
            # Wikipedia API returns a dict with page IDs as keys
            page_id = next(iter(pages))
            extract = pages[page_id].get('extract', "")
            
        if not extract:
            print(f"[error] Could not retrieve text for page: {page_title}")
            return ""
            
        print(f"[wiki] Successfully fetched {len(extract)} characters.")
        return extract
        
    except Exception as e:
        print(f"[error] Wikipedia fetch failed: {e}")
        return ""


def load_corpus(path: str) -> str:
    if path == '/dev/stdin':
        return sys.stdin.read()
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except FileNotFoundError:
        print(f"[warn] corpus file not found: {path!r} — using synthetic fallback")
        return SYNTHETIC_CORPUS


def build_library(
    corpus: Optional[str] = None,
    topic: Optional[str] = None,
    output: str = "",
    steps: int = 100_000,
    min_density: float = 0.3,
    num_initial_patterns: int = 20,
    registry_path: Optional[str] = None,
    name: Optional[str] = None,
    domain: str = "text",
) -> int:
    if topic:
        corpus_text = fetch_wikipedia_content(topic)
        if not corpus_text:
            print("[error] Could not build corpus from topic. Aborting.")
            return 1
        source_label = f"wiki:{topic}"
    elif corpus:
        corpus_text = load_corpus(corpus)
        source_label = corpus
    else:
        print("[error] Either corpus or topic must be specified.")
        return 1

    adapter = CharClassAdapter()
    tokens = [adapter.encode_char(ch) for ch in corpus_text]
    if not tokens:
        print("[error] Empty corpus after encoding")
        return 1

    agent = HPMAgent(num_initial_patterns=num_initial_patterns, obs_dim=5)
    print(f"[init] {len(agent.patterns)} patterns, running {steps} steps")

    for step in range(steps):
        obs = tokens[step % len(tokens)]
        agent.perceive_and_learn(obs)

        if (step + 1) % 10_000 == 0:
            weights = [p.weight for p in agent.patterns]
            field_freq = {p.id: p.weight for p in agent.patterns}
            densities = [
                compute_density(p, agent.obs_buffer, field_freq)
                for p in agent.patterns
            ]
            avg_d = float(np.mean(densities)) if densities else 0.0
            best_w = float(max(weights)) if weights else 0.0
            print(f"[step={step+1}] pop_size={len(agent.patterns)} "
                  f"best_weight={best_w:.3f} avg_density={avg_d:.3f}")

    field_freq = {p.id: p.weight for p in agent.patterns}
    kept = []
    for p in agent.patterns:
        d = compute_density(p, agent.obs_buffer, field_freq)
        if d > min_density and p.weight > 0.01:
            p.source_corpus = source_label
            p.density_at_save = d
            kept.append(p)

    if not kept:
        print("[warn] No patterns passed the density filter. Lowering --min-density may help.")
        return 0

    PatternSerializer.save(kept, output)
    densities = [p.density_at_save for p in kept]
    print(f"\n[done] {len(kept)} patterns saved to {output}")
    print(f"       density: min={min(densities):.3f} "
          f"mean={float(np.mean(densities)):.3f} max={max(densities):.3f}")

    if registry_path:
        registry = LibraryRegistry(registry_path)
        entry_name = name or os.path.splitext(os.path.basename(output))[0]
        registry.upsert(
            name=entry_name,
            path=output,
            domain=domain,
            status="seed",
            bundle_kind="flat",
            level_contract="l1",
            obs_dims=[5],
            source=source_label,
            density_mean=float(np.mean(densities)),
            density_min=float(min(densities)),
            density_max=float(max(densities)),
            pattern_count=len(kept),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        print(f"[registry] registered {entry_name!r} in {registry_path}")

    return 0


def compute_density(p, obs_buffer, field_freq):
    aff = affective_score(p, obs_buffer)
    soc = social_score(p, field_freq)
    # pattern_density expects [ep, aff, soc, field] or similar depending on implementation
    # Checking hpm_ai_v4/evaluators/metrics.py definition:
    # def pattern_density(pattern: HierarchicalPattern, obs_seq: List[int], scores: List[float]) -> float:
    # scores should be [epistemic, affective, social, field]
    from hpm_ai_v4.evaluators.metrics import epistemic_score
    ep = epistemic_score(p)
    # field score is usually soc * gamma_field
    field_infl = 0.2 * soc
    return pattern_density(p, obs_buffer, [ep, aff, soc, field_infl])


def main():
    parser = argparse.ArgumentParser(description="Build HPM pattern library from corpus or topic")
    parser.add_argument('--corpus', help='Path to text corpus')
    parser.add_argument('--topic', help='Wikipedia topic to fetch and learn from')
    parser.add_argument('--output', required=True, help='Output .pkl path')
    parser.add_argument('--steps', type=int, default=100_000)
    parser.add_argument('--min-density', type=float, default=0.3)
    parser.add_argument('--num-initial-patterns', type=int, default=20)
    parser.add_argument('--registry', help='Optional JSON registry path for curated libraries')
    parser.add_argument('--name', help='Registry entry name')
    parser.add_argument('--domain', default='text', help='Registry domain label')
    args = parser.parse_args()
    if not args.topic and not args.corpus:
        print("[error] Either --corpus or --topic must be specified.")
        sys.exit(1)
    sys.exit(
        build_library(
            corpus=args.corpus,
            topic=args.topic,
            output=args.output,
            steps=args.steps,
            min_density=args.min_density,
            num_initial_patterns=args.num_initial_patterns,
            registry_path=args.registry,
            name=args.name,
            domain=args.domain,
        )
    )


if __name__ == '__main__':
    main()
