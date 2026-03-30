"""
NLP child language experiment: semantic category alignment.

Runs the Observer over synthetic child-directed sentence observations and
measures whether learned nodes align with known semantic categories
(animal, family, adult, child_person, food, object, place).

A node with high category purity fired predominantly on one semantic
category — the Observer discovered latent word classes without supervision.

Usage:
    PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_nlp.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn import Observer, calibrate_tau
from hfn.fractal import hausdorff_distance
from hfn.tiered_forest import TieredForest
from hpm_fractal_node.nlp.nlp_loader import (
    generate_sentences, generate_corpus_observations, category_names, D, VOCAB,
)
from hpm_fractal_node.nlp.nlp_world_model import build_nlp_world_model
from hpm_fractal_node.nlp.query_llm import QueryLLM
from hpm_fractal_node.nlp.converter_llm import ConverterLLM
from hpm_fractal_node.nlp.download_corpus import corpus_path, download as download_corpus

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "tinyllama:latest"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SAMPLES = 2000
N_PASSES = 3
SEED = 42

TAU_SIGMA = 1.0
TAU_MARGIN = 5.0

N_CATEGORIES = 7  # animal, adult, child_person, family, food, object, place
RANDOM_BASELINE = 1.0 / N_CATEGORIES  # ~0.143


def purity(counts: dict) -> float:
    """Fraction of observations matching the dominant value."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(counts.values()) / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading {N_SAMPLES} NLP observations (D={D}) ...", flush=True)
    data = generate_sentences(seed=SEED)
    print(f"  {len(data)} synthetic observations loaded")

    # Mix in real corpus observations if available
    _corpus = corpus_path()
    if not _corpus.exists():
        print("  Corpus not found — downloading Peter Rabbit ...", flush=True)
        try:
            download_corpus()
        except SystemExit:
            print("  Corpus download failed — running with synthetic only")
    if _corpus.exists():
        corpus_obs = generate_corpus_observations(_corpus, max_obs=500, seed=SEED)
        data = data + corpus_obs
        print(f"  +{len(corpus_obs)} corpus observations (real text, unknown words)")
    print(f"  {len(data)} total observations")
    cats = category_names()
    print(f"  {len(cats)} semantic categories: {cats}")

    print("\nBuilding NLP world model ...")
    _cold_dir = Path(__file__).parents[2] / "data" / "hfn_nlp_cold"
    _cold_dir.mkdir(parents=True, exist_ok=True)
    forest, prior_ids = build_nlp_world_model(
        forest_cls=TieredForest,
        cold_dir=_cold_dir,
        max_hot=500,
    )
    forest.set_protected(prior_ids)
    print(f"  {len(forest)} priors, {len(prior_ids)} protected")

    tau = calibrate_tau(D, sigma_scale=TAU_SIGMA, margin=TAU_MARGIN)
    print(f"  tau = {tau:.2f}")

    # Wire in QueryLLM via ollama
    _test_query = QueryLLM(host=OLLAMA_HOST, model=OLLAMA_MODEL)
    _test_resp = _test_query._call_ollama("say hi")
    if _test_resp:
        print(f"  Ollama ({OLLAMA_MODEL}) reachable at {OLLAMA_HOST} — QueryLLM active")
        _query = _test_query
        _converter = ConverterLLM()
    else:
        print(f"  Ollama not reachable at {OLLAMA_HOST} — running without QueryLLM")
        _query = None
        _converter = None

    obs = Observer(
        forest,
        tau=tau,
        protected_ids=prior_ids,
        recombination_strategy="nearest_prior",
        hausdorff_absorption_threshold=0.15,
        persistence_guided_absorption=True,
        lacunarity_guided_creation=True,
        lacunarity_creation_radius=0.08,
        multifractal_guided_absorption=True,
        multifractal_crowding_radius=0.12,
        query=_query,
        converter=_converter,
        gap_query_threshold=0.05,
        max_expand_depth=2,
        vocab=VOCAB,
    )

    # ------------------------------------------------------------------
    # Multi-pass observation — track (true_word, category) per node
    # ------------------------------------------------------------------
    node_explanations: dict[str, list[tuple[str, str]]] = defaultdict(list)

    print(f"\nRunning {N_PASSES} passes over {N_SAMPLES} observations ...")
    for p in range(N_PASSES):
        n_explained = 0
        n_unexplained = 0
        rng = np.random.default_rng(SEED + p)
        order = rng.permutation(len(data))

        for i in order:
            vec, true_word, category = data[i]
            x = vec.astype(np.float64)
            # Give QueryLLM the actual word for unknown corpus tokens
            if _query is not None and category == "unknown":
                _query.current_target = true_word
            result = obs.observe(x)
            if _query is not None:
                _query.current_target = None
            # Debug: print accuracy scores for first 3 unknown corpus obs
            if p == 0 and category == "unknown" and (n_explained + n_unexplained) < 10:
                scores = result.accuracy_scores
                best = max(scores.values(), default=0.0) if scores else 0.0
                print(f"    [debug] '{true_word}' best_acc={best:.3f} gap={1-best:.3f} nodes={len(scores)}", flush=True)
            forest._on_observe()
            if (n_explained + n_unexplained) % 500 == 0:
                print(f"    {n_explained + n_unexplained}/{N_SAMPLES} ...", flush=True)

            if result.explanation_tree:
                best_id = max(result.accuracy_scores, key=lambda k: result.accuracy_scores[k])
                node_explanations[best_id].append((true_word, category))
                n_explained += 1
            else:
                n_unexplained += 1

        n_total = len(data)
        print(f"  Pass {p+1}: explained {n_explained}/{n_total} "
              f"({100*n_explained/n_total:.1f}%), "
              f"unexplained {n_unexplained} "
              f"({100*n_unexplained/n_total:.1f}%)")

    # ------------------------------------------------------------------
    # Coverage summary
    # ------------------------------------------------------------------
    prior_explained = sum(
        len(v) for k, v in node_explanations.items() if k in prior_ids
    )
    learned_nodes = [k for k in node_explanations if k not in prior_ids]
    learned_explained = sum(len(node_explanations[k]) for k in learned_nodes)
    total_attributed = prior_explained + learned_explained
    n_obs_total = N_PASSES * N_SAMPLES

    print(f"\n=== Coverage (over all passes, {n_obs_total} total obs) ===")
    print(f"  Prior nodes explained:   {prior_explained:5d} ({100*prior_explained/n_obs_total:.1f}%)")
    print(f"  Learned nodes explained: {learned_explained:5d} ({100*learned_explained/n_obs_total:.1f}%)")
    print(f"  Total attributed:        {total_attributed:5d} ({100*total_attributed/n_obs_total:.1f}%)")
    print(f"  Learned node count:      {len(learned_nodes)}")

    # ------------------------------------------------------------------
    # Prior node category alignment
    # ------------------------------------------------------------------
    print(f"\n=== Prior node category alignment (top priors by coverage) ===")

    prior_rows = [
        (k, node_explanations[k])
        for k in prior_ids
        if len(node_explanations.get(k, [])) > 0
    ]
    prior_rows.sort(key=lambda r: len(r[1]), reverse=True)

    for node_id, labels in prior_rows[:15]:
        n = len(labels)
        cat_counts: dict[str, int] = defaultdict(int)
        for _, cat in labels:
            cat_counts[cat] += 1
        pur = purity(cat_counts)
        dom_cat = max(cat_counts, key=lambda k: cat_counts[k])
        cat_str = " ".join(f"{c}:{cnt}" for c, cnt in sorted(cat_counts.items()))
        print(f"  {node_id:<35s}  n={n:4d}  cat_purity={pur:.2f}  dom={dom_cat}  [{cat_str}]")

    # ------------------------------------------------------------------
    # Learned node category alignment
    # ------------------------------------------------------------------
    if not learned_nodes:
        print("\nNo learned nodes — try increasing N_PASSES or lowering tau_margin.")
        return

    print(f"\n=== Learned node category alignment ({len(learned_nodes)} nodes) ===")

    rows = []
    for node_id in learned_nodes:
        if forest.get(node_id) is None:
            continue  # absorbed during a later pass

        labels = node_explanations[node_id]
        n = len(labels)

        cat_counts: dict[str, int] = defaultdict(int)
        word_counts: dict[str, int] = defaultdict(int)
        for word, cat in labels:
            cat_counts[cat] += 1
            word_counts[word] += 1

        cat_pur = purity(cat_counts)
        word_pur = purity(word_counts)
        dom_cat = max(cat_counts, key=lambda k: cat_counts[k]) if cat_counts else "none"
        dom_word = max(word_counts, key=lambda k: word_counts[k]) if word_counts else "none"

        # Nearest prior by Euclidean distance to prior mus
        node_mu = forest.get(node_id).mu
        best_prior = None
        best_dist = float("inf")
        for pid in prior_ids:
            prior_node = forest.get(pid)
            if prior_node is None:
                continue
            d = float(np.linalg.norm(node_mu - prior_node.mu))
            if d < best_dist:
                best_dist = d
                best_prior = pid

        rows.append({
            "id": node_id,
            "n": n,
            "cat_purity": cat_pur,
            "word_purity": word_pur,
            "dom_cat": dom_cat,
            "dom_word": dom_word,
            "nearest_prior": best_prior,
            "prior_dist": best_dist,
        })

    rows.sort(key=lambda r: r["n"], reverse=True)

    header = (f"  {'Node ID':<20s}  {'n':>4s}  {'cat_pur':>7s}  "
              f"{'word_pur':>8s}  {'dom_cat':<14s}  {'nearest_prior':<30s}  dist")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(
            f"  {r['id']:<20s}  {r['n']:>4d}  {r['cat_purity']:>7.3f}  "
            f"{r['word_purity']:>8.3f}  {r['dom_cat']:<14s}  "
            f"{r['nearest_prior'] or 'none':<30s}  {r['prior_dist']:.3f}"
        )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    cat_purities = [r["cat_purity"] for r in rows if r["n"] >= 5]
    word_purities = [r["word_purity"] for r in rows if r["n"] >= 5]

    print(f"\n=== Category purity summary (nodes with n>=5) ===")
    if cat_purities:
        print(f"  Category purity:  mean={np.mean(cat_purities):.3f}  max={np.max(cat_purities):.3f}")
        print(f"  Word purity:      mean={np.mean(word_purities):.3f}  max={np.max(word_purities):.3f}")
        print(f"  Random baseline (category): {RANDOM_BASELINE:.3f}  (1/{N_CATEGORIES})")
    else:
        print("  Insufficient learned nodes with n>=5 for statistics.")

    # ------------------------------------------------------------------
    # Fractal diagnostics
    # ------------------------------------------------------------------
    prior_node_list = [forest.get(k) for k in prior_ids if forest.get(k) is not None]
    learned_node_list = [n for n in forest.active_nodes() if n.id not in prior_ids]

    if learned_node_list:
        hd = hausdorff_distance(learned_node_list, prior_node_list)
        print(f"\n=== Fractal diagnostics ===")
        print(f"  Learned nodes:              {len(learned_node_list)}")
        print(f"  Hausdorff(learned, priors): {hd:.4f}")

    print(f"\n=== Absorbed nodes: {len(obs.absorbed_ids)} ===")

    # ------------------------------------------------------------------
    # Abstraction metrics
    # ------------------------------------------------------------------
    word_ids = prior_ids  # word_ priors are the leaf level

    # BFS depth from nearest prior
    from collections import deque
    depth_map: dict[str, int] = {pid: 0 for pid in word_ids}
    queue: deque = deque(word_ids)
    visited = set(word_ids)
    while queue:
        nid = queue.popleft()
        node = forest.get(nid)
        if node is None:
            continue
        try:
            children = list(node.children()) if callable(node.children) else list(node.children)
        except Exception:
            children = []
        for child in children:
            if child.id not in visited:
                depth_map[child.id] = depth_map[nid] + 1
                visited.add(child.id)
                queue.append(child.id)

    # Non-prior active nodes
    active_non_prior = [
        n for n in forest.active_nodes() if n.id not in prior_ids
    ]

    # Per-node category coverage
    n_depth2 = 0
    n_cross_cat = 0
    n_stable = 0  # appeared in node_explanations (survived all passes)

    for node in active_non_prior:
        depth = depth_map.get(node.id, 0)
        if depth >= 2:
            n_depth2 += 1

        labels = node_explanations.get(node.id, [])
        cats = {cat for _, cat in labels}
        if len(cats) >= 2:
            n_cross_cat += 1

        if node.id in node_explanations:
            n_stable += 1

    print(f"\n=== Abstraction candidates ({len(active_non_prior)} non-prior nodes) ===")
    print(f"  depth >= 2:              {n_depth2}")
    print(f"  cross-category (>=2):    {n_cross_cat}")
    print(f"  stable (appeared in exp):{n_stable}")
    if _query is not None and hasattr(_query, "_cache"):
        print(f"  LLM queries cached:      {len(_query._cache)}")


if __name__ == "__main__":
    main()
