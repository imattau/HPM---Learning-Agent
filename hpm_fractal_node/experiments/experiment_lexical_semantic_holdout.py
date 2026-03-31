"""
WordNet relation-holdout experiment.

This experiment reuses the WordNet-backed lexical-semantic prior forest, but
masks a slice of relation priors and exercises the model on a broader local
text corpus assembled from repository documentation plus Peter Rabbit.

The goal is to test whether HFN can preserve coverage while learning around
missing semantic bridges instead of relying entirely on the full relation set.

Usage:
    PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_lexical_semantic_holdout.py
"""

from __future__ import annotations

import gc
import re
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import psutil

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
HOLDOUT_RATIO = 0.40

TEXT_MIN_LEN = 24
TEXT_MAX_SEGMENTS_PER_SOURCE = 800
TEXT_FILE_PATTERNS = ("*.md",)
TEXT_SOURCE_SKIP_PARTS = {".git", ".venv", "node_modules", "data", ".claude", ".serena"}


@lru_cache(maxsize=1)
def _broad_corpus_segments() -> tuple[str, ...]:
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

    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    segments: list[str] = []
    for path in unique_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = text.replace("\r", " ")
        text_segments = []
        for raw_segment in re.split(r"(?<=[.!?])\s+", text):
            segment = " ".join(raw_segment.split())
            if len(segment) < TEXT_MIN_LEN:
                continue
            text_segments.append(segment)
        if len(text_segments) > TEXT_MAX_SEGMENTS_PER_SOURCE:
            text_segments = text_segments[:TEXT_MAX_SEGMENTS_PER_SOURCE]
        segments.extend(text_segments)

    if not segments and rabbit.exists():
        segments = [line.strip() for line in rabbit.read_text(encoding="utf-8", errors="ignore").splitlines() if len(line.strip()) >= TEXT_MIN_LEN]

    return tuple(segments)


def _holdout_relation_kept(rel_id: str, holdout_ratio: float) -> bool:
    if holdout_ratio <= 0.0:
        return True
    bucket = base._stable_int(rel_id) % 1000 / 1000.0
    return bucket >= holdout_ratio


def _build_holdout_world_model(holdout_ratio: float):
    inventory = base.build_inventory("large")
    root = Path(__file__).parents[2]
    holdout_tag = f"{int(holdout_ratio * 100):02d}"
    cold_dir = root / "data" / f"hfn_wordnet_holdout_{holdout_tag}"
    cold_dir.mkdir(parents=True, exist_ok=True)
    forest = TieredForest(
        D=D,
        forest_id=f"wordnet_holdout_{holdout_tag}",
        cold_dir=cold_dir,
        max_hot=base.MODE_CONFIG["large"]["max_hot"],
        sweep_every=150,
        min_free_ram_mb=256,
    )

    prior_ids: set[str] = set()
    heldout_relation_ids: set[str] = set()

    def add(node):
        forest.register(node)
        prior_ids.add(node.id)
        return node

    synset_nodes: dict[str, object] = {}
    lemma_nodes: dict[str, object] = {}
    relation_nodes: dict[str, object] = {}

    for lemma_name, text in sorted(inventory.lemma_texts.items()):
        node_id = f"lemma_{lemma_name.replace(' ', '_')}"
        lemma_nodes[lemma_name] = add(base._node(node_id, base._compose_mu("n", 0, text, relation_tag="lemma"), sigma_scale=0.18))

    for name, record in inventory.synset_records.items():
        syn = record.synset
        node = add(base._node(f"syn_{name.replace('.', '_')}", record.mu, sigma_scale=0.24))
        synset_nodes[name] = node
        for lemma in syn.lemmas():
            lemma_name = lemma.name().replace("_", " ")
            lemma_node = lemma_nodes.get(lemma_name)
            if lemma_node is not None:
                node.add_child(lemma_node)

    for src_name, rel_tag, tgt_name in inventory.relation_specs:
        rel_id = f"rel_{rel_tag}_{src_name.replace('.', '_')}__{tgt_name.replace('.', '_')}"
        if not _holdout_relation_kept(rel_id, holdout_ratio):
            heldout_relation_ids.add(rel_id)
            continue
        src = synset_nodes.get(src_name)
        tgt = synset_nodes.get(tgt_name)
        if src is None or tgt is None:
            continue
        if rel_id in relation_nodes:
            continue
        text = f"relation {rel_tag} {src_name.replace('_', ' ')} {tgt_name.replace('_', ' ')}"
        rel_mu = base._compose_mu(
            "n",
            min(inventory.depth_by_synset.get(src_name, 0), inventory.depth_by_synset.get(tgt_name, 0)),
            text,
            relation_tag=rel_tag,
        )
        rel_node = add(base._node(rel_id, rel_mu, sigma_scale=0.32))
        rel_node.add_child(src)
        rel_node.add_child(tgt)
        relation_nodes[rel_id] = rel_node

    pos_roots: dict[str, object] = {}
    pos_groups: dict[str, list] = defaultdict(list)
    lex_groups: dict[str, list] = defaultdict(list)
    depth_groups: dict[str, list] = defaultdict(list)

    for name, node in synset_nodes.items():
        record = inventory.synset_records[name]
        pos_groups[record.synset.pos()].append(node)
        lex_groups[record.label].append(node)
        depth_bucket = next(bucket_name for lo, hi, bucket_name in base.DEPTH_BUCKETS if lo <= record.depth <= hi)
        depth_groups[depth_bucket].append(node)

    for pos, nodes in pos_groups.items():
        pos_id = f"pos_{base.POS_LABEL[base._normalize_pos(pos)]}"
        pos_node = add(base._node(pos_id, np.mean([n.mu for n in nodes], axis=0), sigma_scale=0.4))
        for child in nodes:
            pos_node.add_child(child)
        pos_roots[pos] = pos_node

    lex_roots: dict[str, object] = {}
    for lexname, nodes in lex_groups.items():
        lex_id = f"lex_{lexname.replace('.', '_')}"
        lex_node = add(base._node(lex_id, np.mean([n.mu for n in nodes], axis=0), sigma_scale=0.46))
        for child in nodes:
            lex_node.add_child(child)
        lex_roots[lexname] = lex_node

    depth_roots: dict[str, object] = {}
    for bucket_name, nodes in depth_groups.items():
        depth_node = add(base._node(bucket_name, np.mean([n.mu for n in nodes], axis=0), sigma_scale=0.42))
        for child in nodes:
            depth_node.add_child(child)
        depth_roots[bucket_name] = depth_node

    ontology_root = add(
        base._node(
            "ontology_root",
            np.mean([n.mu for n in list(pos_roots.values()) + list(lex_roots.values()) + list(depth_roots.values())], axis=0),
            sigma_scale=0.6,
        )
    )
    for child in list(pos_roots.values()) + list(lex_roots.values()) + list(depth_roots.values()):
        ontology_root.add_child(child)

    forest.set_protected(prior_ids)
    return forest, prior_ids, inventory, {
        "heldout_relation_ids": heldout_relation_ids,
        "synset_nodes": synset_nodes,
        "lemma_nodes": lemma_nodes,
        "relation_nodes": relation_nodes,
        "pos_roots": pos_roots,
        "lex_roots": lex_roots,
        "depth_roots": depth_roots,
        "ontology_root": ontology_root,
    }


def _generate_observations(inventory, n_samples: int, seed: int):
    segments = _broad_corpus_segments()
    if not segments:
        raise ValueError("No corpus segments were found for the relation-holdout experiment")

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(segments), size=n_samples, replace=True)
    data: list[tuple[np.ndarray, str, str]] = []
    for obs_index, segment_idx in enumerate(idxs):
        sentence = segments[segment_idx]
        tokens = base._tokenize_text(sentence)
        anchor = base._best_corpus_anchor(inventory, tokens, seed=seed + obs_index)
        anchor_text = f"{sentence} anchor {anchor.synset.name()} {anchor.label}"
        vec = base._compose_mu(
            base._normalize_pos(anchor.synset.pos()),
            anchor.depth,
            anchor_text,
            relation_tag=f"corpus:{anchor.label}",
        )
        data.append((vec, sentence, anchor.label))
    return data


def _run_variant(name: str, holdout_ratio: float, data: list[tuple[np.ndarray, str, str]]) -> dict:
    print("\n" + "=" * 72)
    print(f"  Variant: {name}")
    print("\n" + "=" * 72)

    forest, prior_ids, inventory, components = _build_holdout_world_model(holdout_ratio)
    heldout_relation_ids = components["heldout_relation_ids"]

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
    print(f"  Held-out relation priors: {len(heldout_relation_ids)}")
    print(f"  Hot priors on start: {forest.hot_count()}")

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
        rng = np.random.default_rng(SEED + p)
        order = rng.permutation(len(data))
        for i in order:
            vec, _, category = data[i]
            result = obs.observe(vec.astype(np.float64))
            forest._on_observe()
            if result.explanation_tree:
                best_id = max(result.accuracy_scores, key=lambda k: result.accuracy_scores[k])
                node_explanations[best_id].append(category)
                n_explained += 1
            else:
                n_unexplained += 1
            rss = process.memory_info().rss
            if rss > peak_rss:
                peak_rss = rss
        print(f"  Pass {p + 1}: explained {n_explained}/{len(data)} ({100 * n_explained / len(data):.1f}%), unexplained {n_unexplained}")

    elapsed = time.perf_counter() - t0
    gc.collect()
    peak_rss = max(peak_rss, process.memory_info().rss)

    learned_nodes = [k for k in node_explanations if k not in prior_ids]
    prior_explained = sum(len(v) for k, v in node_explanations.items() if k in prior_ids)
    learned_explained = sum(len(node_explanations[k]) for k in learned_nodes)
    total_attributed = prior_explained + learned_explained
    n_obs_total = N_PASSES * len(data)

    layer_counts = defaultdict(int)
    layer_rank_sum = 0.0
    layer_rank_n = 0
    for node_id, labels in node_explanations.items():
        layer = base._layer_of(node_id)
        layer_counts[layer] += len(labels)
        layer_rank_sum += base._layer_rank(layer) * len(labels)
        layer_rank_n += len(labels)

    cat_purities = []
    for node_id in learned_nodes:
        if forest.get(node_id) is None:
            continue
        labels = node_explanations[node_id]
        if len(labels) < 5:
            continue
        cat_counts: dict[str, int] = defaultdict(int)
        for cat in labels:
            cat_counts[cat] += 1
        cat_purities.append(base.purity(cat_counts))

    mean_purity = float(np.mean(cat_purities)) if cat_purities else float("nan")
    mean_layer = layer_rank_sum / layer_rank_n if layer_rank_n else 0.0

    return {
        "variant": name,
        "elapsed_s": elapsed,
        "peak_rss_delta_mb": (peak_rss - start_rss) / (1024 ** 2),
        "n_priors": len(prior_ids),
        "heldout_relation_priors": len(heldout_relation_ids),
        "learned_nodes_surviving": sum(1 for k in learned_nodes if forest.get(k) is not None),
        "learned_nodes_explained": len(learned_nodes),
        "final_node_count": len(forest),
        "coverage_pct": 100.0 * total_attributed / n_obs_total,
        "mean_purity": mean_purity,
        "n_purity_nodes": len(cat_purities),
        "mean_explaining_layer": mean_layer,
        "layer_counts": dict(layer_counts),
        "counts": counts,
        "prior_explained": prior_explained,
        "learned_explained": learned_explained,
        "total_attributed": total_attributed,
        "relation_layer_explanations": layer_counts.get("relation", 0),
    }


def _print_comparison(baseline: dict, holdout: dict) -> None:
    print("\n" + "=" * 72)
    print("  COMPARISON")
    print("\n" + "=" * 72)
    rows = [
        ("Wall-clock time (s)", f"{baseline['elapsed_s']:.2f}", f"{holdout['elapsed_s']:.2f}"),
        ("Peak RSS delta (MB)", f"{baseline['peak_rss_delta_mb']:.1f}", f"{holdout['peak_rss_delta_mb']:.1f}"),
        ("Prior nodes", f"{baseline['n_priors']}", f"{holdout['n_priors']}"),
        ("Held-out relation priors", f"{baseline['heldout_relation_priors']}", f"{holdout['heldout_relation_priors']}"),
        ("Final node count", f"{baseline['final_node_count']}", f"{holdout['final_node_count']}"),
        ("Learned nodes surviving", f"{baseline['learned_nodes_surviving']}", f"{holdout['learned_nodes_surviving']}"),
        ("Learned nodes explained", f"{baseline['learned_nodes_explained']}", f"{holdout['learned_nodes_explained']}"),
        ("Coverage %", f"{baseline['coverage_pct']:.2f}%", f"{holdout['coverage_pct']:.2f}%"),
        ("Relation-layer explanations", f"{baseline['relation_layer_explanations']}", f"{holdout['relation_layer_explanations']}"),
        ("Mean category purity (n>=5)", f"{baseline['mean_purity']:.3f}", f"{holdout['mean_purity']:.3f}"),
        ("Purity-eligible nodes", f"{baseline['n_purity_nodes']}", f"{holdout['n_purity_nodes']}"),
        ("Mean explaining layer", f"{baseline['mean_explaining_layer']:.2f}", f"{holdout['mean_explaining_layer']:.2f}"),
    ]
    print(f"  {'Metric':<30} {'Baseline':>14} {'Holdout':>14}")
    print("  " + "-" * 60)
    for metric, left, right in rows:
        print(f"  {metric:<30} {left:>14} {right:>14}")

    for key in ("lemma", "synset", "relation", "pos", "lexname", "depth", "ontology", "learned"):
        b = baseline["layer_counts"].get(key, 0)
        h = holdout["layer_counts"].get(key, 0)
        print(f"  layer[{key:<10}] {b:>10} {h:>10}")

    if baseline["coverage_pct"] > 0:
        print(f"\n  Coverage delta: {holdout['coverage_pct'] - baseline['coverage_pct']:+.4f}%")
    if baseline["peak_rss_delta_mb"] > 0:
        saving = (1.0 - holdout["peak_rss_delta_mb"] / max(baseline["peak_rss_delta_mb"], 1e-9)) * 100.0
        print(f"  Peak RSS saving: {saving:+.1f}%")


def main() -> None:
    segments = _broad_corpus_segments()
    print("Relation-holdout WordNet experiment")
    print(f"  D={D}, N_SAMPLES={N_SAMPLES}, N_PASSES={N_PASSES}, SEED={SEED}")
    print("  Observation stream is sampled from a broader local text corpus and grounded in WordNet.")
    print(f"  Corpus segments available: {len(segments)}")

    inventory = base.build_inventory("large")
    data = _generate_observations(inventory, N_SAMPLES, seed=SEED)
    print(f"  Generated {len(data)} corpus observations")

    baseline = _run_variant("baseline", 0.0, data)
    holdout = _run_variant("holdout", HOLDOUT_RATIO, data)
    _print_comparison(baseline, holdout)


if __name__ == "__main__":
    main()
