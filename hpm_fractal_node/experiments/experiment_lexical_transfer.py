"""
WordNet lexical transfer experiment.

This experiment reuses the WordNet-backed lexical-semantic prior forest and
compares two observation streams:
  - in-domain text from Peter Rabbit plus repository markdown
  - out-of-domain vocabulary sampled from ``nltk.corpus.words`` after removing
    tokens already covered by the WordNet lemma inventory

The goal is to test whether the same external ontology can keep explaining
observations when surface vocabulary moves outside the direct prior anchors.

Usage:
    PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_lexical_transfer.py
"""

from __future__ import annotations

import gc
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import nltk
import numpy as np
import psutil
from nltk.corpus import words as nltk_words

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn import Observer, calibrate_tau
from hfn.tiered_forest import TieredForest
from hpm_fractal_node.experiments import experiment_lexical_semantic_forest as base
from hpm_fractal_node.nlp.download_corpus import corpus_path, download as download_corpus

D = base.D
SEED = base.SEED
N_SAMPLES = 1200
N_PASSES = 2
TAU_SIGMA = base.TAU_SIGMA
TAU_MARGIN = base.TAU_MARGIN

TEXT_MIN_LEN = 24
TEXT_MAX_SEGMENTS_PER_SOURCE = 800
TEXT_FILE_PATTERNS = ("*.md",)
TEXT_SOURCE_SKIP_PARTS = {".git", ".venv", "node_modules", "data", ".claude", ".serena"}
OOD_TOKEN_MIN_LEN = 4
OOD_MAX_TOKEN_POOL = 50000
OOD_TOKENS_PER_OBS = (5, 9)


@dataclass(frozen=True)
class ObservationBatch:
    observations: list[tuple[np.ndarray, str, str]]
    known_token_hits: int
    fallback_obs: int
    token_total: int
    unique_tokens: int
    source_name: str


def _normalize_token(token: str) -> str:
    helper = getattr(base, "_normalize_token", None)
    if callable(helper):
        return helper(token)
    token = token.lower().replace("'", "")
    return re.sub(r"[^a-z]+", "", token)


def _tokenize_text(text: str) -> list[str]:
    helper = getattr(base, "_tokenize_text", None)
    if callable(helper):
        return list(helper(text))
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z'-]*", text):
        token = _normalize_token(raw)
        if token:
            tokens.append(token)
    return tokens


def _stable_int(text: str) -> int:
    helper = getattr(base, "_stable_int", None)
    if callable(helper):
        return int(helper(text))
    return abs(hash(text))


def _ensure_words_corpus() -> None:
    try:
        nltk.data.find("corpora/words")
    except LookupError as exc:
        raise LookupError(
            "NLTK words corpus not found. Install it with: nltk.download('words')"
        ) from exc


@lru_cache(maxsize=1)
def _in_domain_segments() -> tuple[str, ...]:
    root = Path(__file__).parents[2]
    paths: list[Path] = []

    rabbit = corpus_path()
    if not rabbit.exists():
        try:
            download_corpus()
        except SystemExit:
            pass
    if rabbit.exists():
        paths.append(rabbit)

    for pattern in TEXT_FILE_PATTERNS:
        for path in sorted(root.rglob(pattern)):
            if any(part in TEXT_SOURCE_SKIP_PARTS or part.startswith(".") for part in path.parts):
                continue
            paths.append(path)

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    segments: list[str] = []
    for path in unique_paths:
        text = path.read_text(encoding="utf-8", errors="ignore").replace("\r", " ")
        text_segments: list[str] = []
        for raw_segment in re.split(r"(?<=[.!?])\s+", text):
            segment = " ".join(raw_segment.split())
            if len(segment) < TEXT_MIN_LEN:
                continue
            text_segments.append(segment)
        if len(text_segments) > TEXT_MAX_SEGMENTS_PER_SOURCE:
            text_segments = text_segments[:TEXT_MAX_SEGMENTS_PER_SOURCE]
        segments.extend(text_segments)

    if not segments and rabbit.exists():
        segments = [
            line.strip()
            for line in rabbit.read_text(encoding="utf-8", errors="ignore").splitlines()
            if len(line.strip()) >= TEXT_MIN_LEN
        ]

    return tuple(segments)


def _fallback_anchor(inventory, tokens: list[str], seed: int):
    seeds = getattr(inventory, "seed_synsets", ())
    if not seeds:
        raise ValueError("Inventory does not expose seed_synsets for fallback anchoring")
    index = _stable_int(f"{seed}:{' '.join(tokens)}") % len(seeds)
    syn = seeds[index]
    return inventory.synset_records[syn.name()]


def _best_corpus_anchor_with_info(inventory, tokens: list[str], seed: int):
    helper = getattr(base, "_best_corpus_anchor", None)
    if callable(helper):
        call_patterns = (
            lambda: helper(inventory, tokens, seed=seed),
            lambda: helper(inventory, tokens, seed),
            lambda: helper(tokens, inventory, seed=seed),
            lambda: helper(tokens, inventory, seed),
        )
        for call in call_patterns:
            try:
                result = call()
                if result is not None:
                    return result, False
            except TypeError:
                continue

    candidate_counts: dict[str, int] = defaultdict(int)
    lemma_to_synsets = getattr(inventory, "lemma_to_synsets", {})
    depth_by_synset = getattr(inventory, "depth_by_synset", {})
    for token in tokens:
        for syn_name in lemma_to_synsets.get(token, ()):
            candidate_counts[syn_name] += 1

    if candidate_counts:
        best_name = max(
            candidate_counts,
            key=lambda name: (
                candidate_counts[name],
                -depth_by_synset.get(name, 0),
                name,
            ),
        )
        return inventory.synset_records[best_name], False

    return _fallback_anchor(inventory, tokens, seed), True


def _compose_observation_mu(anchor, text: str, relation_tag: str) -> np.ndarray:
    pos = base._normalize_pos(anchor.synset.pos())
    compose = getattr(base, "_compose_mu", None)
    if callable(compose):
        call_patterns = (
            lambda: compose(pos, anchor.depth, text, relation_tag=relation_tag),
            lambda: compose(pos, anchor.depth, text, relation_tag),
            lambda: compose(anchor.synset.pos(), anchor.depth, text, relation_tag=relation_tag),
            lambda: compose(anchor.synset.pos(), anchor.depth, text, relation_tag),
        )
        for call in call_patterns:
            try:
                vec = call()
                if vec is not None:
                    return vec
            except TypeError:
                continue
    return base._compose_mu(pos, anchor.depth, text, relation_tag=relation_tag)


def _ood_token_pool(inventory) -> tuple[str, ...]:
    _ensure_words_corpus()
    known = set(getattr(inventory, "lemma_to_synsets", {}))
    pool: list[str] = []
    seen: set[str] = set()
    for raw in nltk_words.words():
        token = _normalize_token(raw)
        if len(token) < OOD_TOKEN_MIN_LEN:
            continue
        if not token.isalpha():
            continue
        if token in known or token in seen:
            continue
        seen.add(token)
        pool.append(token)
        if len(pool) >= OOD_MAX_TOKEN_POOL:
            break
    return tuple(pool)


def _make_in_domain_batch(inventory, n_samples: int, seed: int) -> ObservationBatch:
    segments = _in_domain_segments()
    if not segments:
        raise ValueError("No in-domain corpus segments were found for the lexical transfer experiment")

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(segments), size=n_samples, replace=True)

    observations: list[tuple[np.ndarray, str, str]] = []
    known_token_hits = 0
    fallback_obs = 0
    token_total = 0
    unique_tokens: set[str] = set()

    for obs_index, segment_index in enumerate(indices):
        sentence = segments[segment_index]
        tokens = _tokenize_text(sentence)
        unique_tokens.update(tokens)
        token_total += len(tokens)
        known_token_hits += sum(1 for token in tokens if token in inventory.lemma_to_synsets)
        anchor, used_fallback = _best_corpus_anchor_with_info(inventory, tokens, seed=seed + obs_index)
        fallback_obs += int(used_fallback)
        anchor_text = f"{sentence} anchor {anchor.synset.name()} {anchor.label}"
        vec = _compose_observation_mu(anchor, anchor_text, relation_tag=f"corpus:{anchor.label}")
        observations.append((vec, sentence, anchor.label))

    return ObservationBatch(
        observations=observations,
        known_token_hits=known_token_hits,
        fallback_obs=fallback_obs,
        token_total=token_total,
        unique_tokens=len(unique_tokens),
        source_name="in_domain",
    )


def _make_ood_batch(inventory, token_pool: tuple[str, ...], n_samples: int, seed: int) -> ObservationBatch:
    pool = token_pool
    if not pool:
        raise ValueError("No out-of-domain tokens were found for the lexical transfer experiment")

    rng = np.random.default_rng(seed)
    observations: list[tuple[np.ndarray, str, str]] = []
    token_total = 0
    unique_tokens: set[str] = set()

    min_tokens, max_tokens = OOD_TOKENS_PER_OBS
    for obs_index in range(n_samples):
        n_tokens = int(rng.integers(min_tokens, max_tokens + 1))
        replace = len(pool) < n_tokens
        tokens = list(rng.choice(pool, size=n_tokens, replace=replace))
        unique_tokens.update(tokens)
        token_total += len(tokens)
        anchor, _ = _best_corpus_anchor_with_info(inventory, tokens, seed=seed + obs_index)
        pseudo_sentence = " ".join(tokens)
        anchor_text = f"{pseudo_sentence} anchor {anchor.synset.name()} {anchor.label}"
        vec = _compose_observation_mu(anchor, anchor_text, relation_tag=f"ood:{anchor.label}")
        observations.append((vec, pseudo_sentence, f"ood:{anchor.label}"))

    return ObservationBatch(
        observations=observations,
        known_token_hits=0,
        fallback_obs=len(observations),
        token_total=token_total,
        unique_tokens=len(unique_tokens),
        source_name="ood_words",
    )


def _build_world_model(mode: str = "large", name: str = "variant"):
    cold_dir = Path(__file__).parents[2] / "data" / f"hfn_wordnet_cold_{mode}"
    cold_dir.mkdir(parents=True, exist_ok=True)
    forest, prior_ids, inventory, _components = base._build_world_model(
        mode,
        forest_cls=TieredForest,
        cold_dir=cold_dir,
        max_hot=base.MODE_CONFIG[mode]["max_hot"],
        sweep_every=150,
        min_free_ram_mb=256,
    )
    try:
        forest.set_protected(prior_ids)
    except Exception:
        pass
    return forest, prior_ids, inventory


def _run_variant(name: str, batch: ObservationBatch, inventory, mode: str) -> dict:
    print("\n" + "=" * 72)
    print(f"  Variant: {name}")
    print("=" * 72)

    forest, prior_ids, _inventory = _build_world_model(mode, name)
    counts = {
        "lemmas": sum(1 for pid in prior_ids if pid.startswith("lemma_")),
        "synsets": sum(1 for pid in prior_ids if pid.startswith("syn_")),
        "relations": sum(1 for pid in prior_ids if pid.startswith("rel_")),
        "pos": sum(1 for pid in prior_ids if pid.startswith("pos_")),
        "lex": sum(1 for pid in prior_ids if pid.startswith("lex_")),
        "depth": sum(1 for pid in prior_ids if pid.startswith("depth_")),
        "ontology": sum(1 for pid in prior_ids if pid == "ontology_root"),
    }
    print(
        f"  Priors: total={len(prior_ids)}  lemmas={counts['lemmas']}  synsets={counts['synsets']}  "
        f"relations={counts['relations']}  pos={counts['pos']}  lex={counts['lex']}  "
        f"depth={counts['depth']}  ontology={counts['ontology']}"
    )
    print(f"  Hot priors on start: {forest.hot_count()}")
    print(f"  Known token rate: {batch.known_token_hits / max(batch.token_total, 1):.2%}")
    print(f"  Fallback rate: {batch.fallback_obs / max(len(batch.observations), 1):.2%}")
    print(f"  Avg tokens / obs: {batch.token_total / max(len(batch.observations), 1):.2f}")
    print(f"  Unique tokens: {batch.unique_tokens}")

    tau = calibrate_tau(D, sigma_scale=TAU_SIGMA, margin=TAU_MARGIN)
    print(f"  tau = {tau:.2f}")

    obs = Observer(
        forest,
        tau=tau,
        protected_ids=prior_ids,
        recombination_strategy="nearest_prior",
        hausdorff_absorption_threshold=0.35,
        hausdorff_absorption_weight_floor=0.4,
        absorption_miss_threshold=18,
        persistence_guided_absorption=True,
        lacunarity_guided_creation=True,
        lacunarity_creation_radius=0.08,
        multifractal_guided_absorption=False,
        gap_query_threshold=None,
        max_expand_depth=2,
        node_use_diag=True,
    )

    node_explanations: dict[str, list[str]] = defaultdict(list)
    process = psutil.Process()
    gc.collect()
    start_rss = process.memory_info().rss
    peak_rss = start_rss
    t0 = time.perf_counter()

    for p in range(N_PASSES):
        n_explained = 0
        n_unexplained = 0
        order = np.random.default_rng(SEED + p).permutation(len(batch.observations))
        for idx in order:
            vec, _, category = batch.observations[idx]
            result = obs.observe(vec.astype(np.float64))
            forest._on_observe()
            if result.explanation_tree:
                best_id = max(result.accuracy_scores, key=lambda key: result.accuracy_scores[key])
                node_explanations[best_id].append(category)
                n_explained += 1
            else:
                n_unexplained += 1
            peak_rss = max(peak_rss, process.memory_info().rss)
        print(
            f"  Pass {p + 1}: explained {n_explained}/{len(batch.observations)} "
            f"({100.0 * n_explained / len(batch.observations):.1f}%), unexplained {n_unexplained}"
        )

    elapsed = time.perf_counter() - t0
    gc.collect()
    peak_rss = max(peak_rss, process.memory_info().rss)

    learned_nodes = [node_id for node_id in node_explanations if node_id not in prior_ids]
    prior_explained = sum(len(labels) for node_id, labels in node_explanations.items() if node_id in prior_ids)
    learned_explained = sum(len(node_explanations[node_id]) for node_id in learned_nodes)
    total_attributed = prior_explained + learned_explained
    n_obs_total = N_PASSES * len(batch.observations)

    layer_counts: dict[str, int] = defaultdict(int)
    layer_rank_sum = 0.0
    layer_rank_n = 0
    for node_id, labels in node_explanations.items():
        layer = base._layer_of(node_id)
        layer_counts[layer] += len(labels)
        layer_rank_sum += base._layer_rank(layer) * len(labels)
        layer_rank_n += len(labels)

    cat_purities: list[float] = []
    for node_id in learned_nodes:
        if forest.get(node_id) is None:
            continue
        labels = node_explanations[node_id]
        if len(labels) < 5:
            continue
        cat_counts: dict[str, int] = defaultdict(int)
        for category in labels:
            cat_counts[category] += 1
        cat_purities.append(base.purity(cat_counts))

    return {
        "variant": name,
        "source_name": batch.source_name,
        "elapsed_s": elapsed,
        "peak_rss_delta_mb": (peak_rss - start_rss) / (1024 ** 2),
        "n_priors": len(prior_ids),
        "known_token_rate": batch.known_token_hits / max(batch.token_total, 1),
        "fallback_rate": batch.fallback_obs / max(len(batch.observations), 1),
        "avg_tokens_per_obs": batch.token_total / max(len(batch.observations), 1),
        "unique_tokens": batch.unique_tokens,
        "learned_nodes_surviving": sum(1 for node_id in learned_nodes if forest.get(node_id) is not None),
        "learned_nodes_explained": len(learned_nodes),
        "final_node_count": len(forest),
        "coverage_pct": 100.0 * total_attributed / max(n_obs_total, 1),
        "mean_purity": float(np.mean(cat_purities)) if cat_purities else float("nan"),
        "n_purity_nodes": len(cat_purities),
        "mean_explaining_layer": (layer_rank_sum / layer_rank_n) if layer_rank_n else 0.0,
        "layer_counts": dict(layer_counts),
        "counts": counts,
        "prior_explained": prior_explained,
        "learned_explained": learned_explained,
        "total_attributed": total_attributed,
        "lemma_layer_explanations": layer_counts.get("lemma", 0),
    }


def _print_comparison(in_domain: dict, ood: dict) -> None:
    print("\n" + "=" * 72)
    print("  COMPARISON")
    print("=" * 72)
    rows = [
        ("Wall-clock time (s)", f"{in_domain['elapsed_s']:.2f}", f"{ood['elapsed_s']:.2f}"),
        ("Peak RSS delta (MB)", f"{in_domain['peak_rss_delta_mb']:.1f}", f"{ood['peak_rss_delta_mb']:.1f}"),
        ("Prior nodes", f"{in_domain['n_priors']}", f"{ood['n_priors']}"),
        ("Known token rate", f"{in_domain['known_token_rate']:.2%}", f"{ood['known_token_rate']:.2%}"),
        ("Fallback rate", f"{in_domain['fallback_rate']:.2%}", f"{ood['fallback_rate']:.2%}"),
        ("Avg tokens / obs", f"{in_domain['avg_tokens_per_obs']:.2f}", f"{ood['avg_tokens_per_obs']:.2f}"),
        ("Unique tokens", f"{in_domain['unique_tokens']}", f"{ood['unique_tokens']}"),
        ("Final node count", f"{in_domain['final_node_count']}", f"{ood['final_node_count']}"),
        ("Learned nodes surviving", f"{in_domain['learned_nodes_surviving']}", f"{ood['learned_nodes_surviving']}"),
        ("Learned nodes explained", f"{in_domain['learned_nodes_explained']}", f"{ood['learned_nodes_explained']}"),
        ("Coverage %", f"{in_domain['coverage_pct']:.2f}%", f"{ood['coverage_pct']:.2f}%"),
        ("Lemma-layer explanations", f"{in_domain['lemma_layer_explanations']}", f"{ood['lemma_layer_explanations']}"),
        ("Mean category purity (n>=5)", f"{in_domain['mean_purity']:.3f}", f"{ood['mean_purity']:.3f}"),
        ("Purity-eligible nodes", f"{in_domain['n_purity_nodes']}", f"{ood['n_purity_nodes']}"),
        ("Mean explaining layer", f"{in_domain['mean_explaining_layer']:.2f}", f"{ood['mean_explaining_layer']:.2f}"),
    ]
    print(f"  {'Metric':<30} {'In-domain':>14} {'OOD':>14}")
    print("  " + "-" * 60)
    for metric, left, right in rows:
        print(f"  {metric:<30} {left:>14} {right:>14}")

    for key in ("lemma", "synset", "relation", "pos", "lexname", "depth", "ontology", "learned"):
        in_value = in_domain["layer_counts"].get(key, 0)
        ood_value = ood["layer_counts"].get(key, 0)
        print(f"  layer[{key:<10}] {in_value:>10} {ood_value:>10}")

    print(f"\n  Coverage delta: {ood['coverage_pct'] - in_domain['coverage_pct']:+.4f}%")
    if in_domain["peak_rss_delta_mb"] > 0:
        saving = (1.0 - ood["peak_rss_delta_mb"] / max(in_domain["peak_rss_delta_mb"], 1e-9)) * 100.0
        print(f"  Peak RSS saving: {saving:+.1f}%")


def main() -> None:
    print("WordNet lexical transfer experiment")
    print(f"  D={D}, N_SAMPLES={N_SAMPLES}, N_PASSES={N_PASSES}, SEED={SEED}")
    print("  Observation stream compares in-domain Peter Rabbit / repo text against out-of-domain words-corpus tokens.")

    inventory = base.build_inventory("large")
    mode = "compact" if N_SAMPLES <= 5 else "large"
    segments = _in_domain_segments()
    ood_pool = _ood_token_pool(inventory)
    print(f"  In-domain segments available: {len(segments)}")
    print(f"  OOD token pool size: {len(ood_pool)}")
    print(f"  World-model mode: {mode}")

    in_domain_batch = _make_in_domain_batch(inventory, N_SAMPLES, seed=SEED)
    ood_batch = _make_ood_batch(inventory, ood_pool, N_SAMPLES, seed=SEED)
    print(f"  Generated {len(in_domain_batch.observations)} in-domain observations")
    print(f"  Generated {len(ood_batch.observations)} OOD observations")

    in_domain = _run_variant("in_domain", in_domain_batch, inventory, mode)
    ood = _run_variant("ood", ood_batch, inventory, mode)
    _print_comparison(in_domain, ood)


if __name__ == "__main__":
    main()
