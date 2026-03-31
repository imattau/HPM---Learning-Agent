"""WordNet lexical curriculum experiment.

This experiment reuses the WordNet-backed lexical-semantic prior forest and
stages progressively harder text streams through the same persistent model:
  - in-domain Peter Rabbit + repository markdown
  - medium out-of-domain vocabulary from ``nltk.corpus.words``
  - hard out-of-domain vocabulary with a stricter lexical filter

The goal is to see whether the forest stretches itself over time: do earlier
stages leave behind reusable learned structure, and does later exposure push
the model toward more abstract explanations?
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
from hpm_fractal_node.experiments import experiment_lexical_transfer as transfer

D = base.D
SEED = base.SEED
N_SAMPLES = 400
N_PASSES = 1
TAU_SIGMA = base.TAU_SIGMA
TAU_MARGIN = base.TAU_MARGIN

OOD_HARD_CHARS = frozenset("qxjzv")
OOD_MILD_TOKEN_MIN_LEN = 5
OOD_MILD_MAX_TOKEN_POOL = 40000
OOD_MILD_TOKENS_PER_OBS = (7, 12)
OOD_HARD_TOKEN_MIN_LEN = 8
OOD_HARD_MAX_TOKEN_POOL = 25000
OOD_HARD_TOKENS_PER_OBS = (10, 16)
TEXT_MIN_LEN = 24
TEXT_MAX_SEGMENTS_PER_SOURCE = 800
TEXT_FILE_PATTERNS = ("*.md",)
TEXT_SOURCE_SKIP_PARTS = {".git", ".venv", "node_modules", "data", ".claude", ".serena"}


@dataclass(frozen=True)
class ObservationBatch:
    observations: list[tuple[np.ndarray, str, str]]
    known_token_hits: int
    fallback_obs: int
    token_total: int
    unique_tokens: int
    source_name: str


@dataclass(frozen=True)
class StageStats:
    stage_name: str
    difficulty: str
    wall_clock_s: float
    rss_delta_mb: float
    prior_nodes: int
    final_node_count: int
    learned_nodes_surviving: int
    new_learned_nodes_surviving: int
    learned_nodes_explained: int
    coverage_pct: float
    mean_explaining_layer: float
    layer_counts: dict[str, int]
    known_token_rate: float
    fallback_rate: float
    avg_tokens: float
    unique_tokens: int
    mean_purity: float
    n_purity_nodes: int


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
    return transfer._in_domain_segments()


def _ood_token_pool(
    inventory,
    *,
    min_len: int,
    max_pool: int,
    require_hard_chars: bool,
) -> tuple[str, ...]:
    _ensure_words_corpus()
    known = set(getattr(inventory, "lemma_to_synsets", {}))
    in_domain_tokens = set()
    for segment in _in_domain_segments():
        in_domain_tokens.update(_tokenize_text(segment))

    pool: list[str] = []
    seen: set[str] = set()
    for raw in nltk_words.words():
        token = _normalize_token(raw)
        if len(token) < min_len:
            continue
        if not token.isalpha():
            continue
        if token in known or token in seen or token in in_domain_tokens:
            continue
        if require_hard_chars and not any(ch in OOD_HARD_CHARS for ch in token):
            continue
        if token.endswith(("ing", "ed", "ly")):
            continue
        seen.add(token)
        pool.append(token)
        if len(pool) >= max_pool:
            break

    if len(pool) < 1000:
        for raw in nltk_words.words():
            token = _normalize_token(raw)
            if len(token) < min_len or not token.isalpha():
                continue
            if token in known or token in seen or token in in_domain_tokens:
                continue
            seen.add(token)
            pool.append(token)
            if len(pool) >= max_pool:
                break

    return tuple(pool)


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


def _make_in_domain_batch(inventory, n_samples: int, seed: int) -> ObservationBatch:
    segments = _in_domain_segments()
    if not segments:
        raise ValueError("No in-domain corpus segments were found for the lexical curriculum experiment")

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


def _make_ood_batch(
    inventory,
    n_samples: int,
    seed: int,
    *,
    token_pool: tuple[str, ...],
    tokens_per_obs: tuple[int, int],
    source_name: str,
) -> ObservationBatch:
    if not token_pool:
        raise ValueError("No out-of-domain tokens were found for the lexical curriculum experiment")

    rng = np.random.default_rng(seed)
    observations: list[tuple[np.ndarray, str, str]] = []
    token_total = 0
    unique_tokens: set[str] = set()

    min_tokens, max_tokens = tokens_per_obs
    for obs_index in range(n_samples):
        n_tokens = int(rng.integers(min_tokens, max_tokens + 1))
        replace = len(token_pool) < n_tokens
        tokens = list(rng.choice(token_pool, size=n_tokens, replace=replace))
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
        source_name=source_name,
    )


def _build_world_model(mode: str = "large"):
    forest, prior_ids, inventory = transfer._build_world_model(mode, "curriculum")
    try:
        forest.set_protected(prior_ids)
    except Exception:
        pass
    return forest, prior_ids, inventory


def _forest_node_ids(forest: TieredForest) -> set[str]:
    mu_index = getattr(forest, "_mu_index", None)
    if isinstance(mu_index, dict):
        return set(mu_index.keys())
    return set()


def _make_observer(forest: TieredForest, prior_ids: set[str]) -> Observer:
    tau = calibrate_tau(D, sigma_scale=TAU_SIGMA, margin=TAU_MARGIN)
    return Observer(
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


def _run_stage(
    stage_name: str,
    difficulty: str,
    batch: ObservationBatch,
    forest: TieredForest,
    prior_ids: set[str],
    observer: Observer,
) -> StageStats:
    print("\n" + "=" * 72)
    print(f"  Stage: {stage_name} ({difficulty})")
    print("=" * 72)

    start_ids = _forest_node_ids(forest)
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
    print(f"  tau = {observer.tau:.2f}")

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
            result = observer.observe(vec.astype(np.float64))
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

    end_ids = _forest_node_ids(forest)
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

    return StageStats(
        stage_name=stage_name,
        difficulty=difficulty,
        wall_clock_s=elapsed,
        rss_delta_mb=(peak_rss - start_rss) / (1024 ** 2),
        prior_nodes=len(prior_ids),
        final_node_count=len(forest),
        learned_nodes_surviving=sum(1 for node_id in end_ids if node_id not in prior_ids),
        new_learned_nodes_surviving=len([node_id for node_id in end_ids - start_ids if node_id not in prior_ids]),
        learned_nodes_explained=len(learned_nodes),
        coverage_pct=100.0 * total_attributed / max(n_obs_total, 1),
        mean_explaining_layer=(layer_rank_sum / layer_rank_n) if layer_rank_n else 0.0,
        layer_counts=dict(layer_counts),
        known_token_rate=batch.known_token_hits / max(batch.token_total, 1),
        fallback_rate=batch.fallback_obs / max(len(batch.observations), 1),
        avg_tokens=batch.token_total / max(len(batch.observations), 1),
        unique_tokens=batch.unique_tokens,
        mean_purity=float(np.mean(cat_purities)) if cat_purities else float("nan"),
        n_purity_nodes=len(cat_purities),
    )


def _print_curriculum_summary(stages: list[StageStats]) -> None:
    print("\n" + "=" * 72)
    print("  CURRICULUM SUMMARY")
    print("=" * 72)
    print(f"  {'Stage':<24}{'Difficulty':<18}{'Coverage':>12}{'New learned':>14}{'Surviving':>12}{'Mean layer':>12}")
    print("  " + "-" * 94)
    for stage in stages:
        print(
            f"  {stage.stage_name:<24}{stage.difficulty:<18}"
            f"{stage.coverage_pct:>11.2f}%"
            f"{stage.new_learned_nodes_surviving:>14}"
            f"{stage.learned_nodes_surviving:>12}"
            f"{stage.mean_explaining_layer:>12.2f}"
        )

    print("\n  Stage details")
    for stage in stages:
        print(
            f"  {stage.stage_name:<24}"
            f"known={stage.known_token_rate:.2%} "
            f"fallback={stage.fallback_rate:.2%} "
            f"avg_tokens={stage.avg_tokens:.2f} "
            f"unique={stage.unique_tokens} "
            f"learned_explained={stage.learned_nodes_explained} "
            f"purity_nodes={stage.n_purity_nodes} "
            f"mean_purity={stage.mean_purity:.3f} "
            f"rss_delta={stage.rss_delta_mb:.1f}MB"
        )
        for key in ("lemma", "synset", "relation", "pos", "lexname", "depth", "ontology", "learned"):
            print(f"    layer[{key:<10}] {stage.layer_counts.get(key, 0):>10}")

    if stages:
        first = stages[0]
        last = stages[-1]
        print(f"\n  Coverage delta (last-first): {last.coverage_pct - first.coverage_pct:+.4f}%")
        print(f"  Learned nodes delta (last-first): {last.learned_nodes_surviving - first.learned_nodes_surviving:+d}")
        print(f"  Mean explaining-layer delta (last-first): {last.mean_explaining_layer - first.mean_explaining_layer:+.2f}")


def main() -> None:
    print("WordNet lexical curriculum experiment")
    print(f"  D={D}, N_SAMPLES={N_SAMPLES}, N_PASSES={N_PASSES}, SEED={SEED}")
    print("  Observation stream stretches a single WordNet forest across progressively harder text streams.")

    mode = "compact" if N_SAMPLES <= 5 else "large"
    forest, prior_ids, inventory = _build_world_model(mode)
    segments = _in_domain_segments()
    mild_pool = _ood_token_pool(
        inventory,
        min_len=OOD_MILD_TOKEN_MIN_LEN,
        max_pool=OOD_MILD_MAX_TOKEN_POOL,
        require_hard_chars=False,
    )
    hard_pool = _ood_token_pool(
        inventory,
        min_len=OOD_HARD_TOKEN_MIN_LEN,
        max_pool=OOD_HARD_MAX_TOKEN_POOL,
        require_hard_chars=True,
    )
    print(f"  In-domain segments available: {len(segments)}")
    print(f"  Mild OOD token pool size: {len(mild_pool)}")
    print(f"  Hard OOD token pool size: {len(hard_pool)}")
    print(f"  World-model mode: {mode}")

    observer = _make_observer(forest, prior_ids)

    stage_batches = [
        ("seed", "easy", transfer._make_in_domain_batch(inventory, N_SAMPLES, seed=SEED)),
        (
            "stretch",
            "medium",
            _make_ood_batch(
                inventory,
                N_SAMPLES,
                seed=SEED + 1,
                token_pool=mild_pool,
                tokens_per_obs=OOD_MILD_TOKENS_PER_OBS,
                source_name="mild_ood",
            ),
        ),
        (
            "stretch_hard",
            "hard",
            _make_ood_batch(
                inventory,
                N_SAMPLES,
                seed=SEED + 2,
                token_pool=hard_pool,
                tokens_per_obs=OOD_HARD_TOKENS_PER_OBS,
                source_name="hard_ood",
            ),
        ),
    ]
    print(f"  Generated {len(stage_batches[0][2].observations)} stage-1 observations")
    print(f"  Generated {len(stage_batches[1][2].observations)} stage-2 observations")
    print(f"  Generated {len(stage_batches[2][2].observations)} stage-3 observations")

    stages: list[StageStats] = []
    for stage_name, difficulty, batch in stage_batches:
        stages.append(_run_stage(stage_name, difficulty, batch, forest, prior_ids, observer))
    _print_curriculum_summary(stages)


if __name__ == "__main__":
    main()
