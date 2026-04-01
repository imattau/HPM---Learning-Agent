"""WordNet lexical cross-stream replay experiment.

This experiment reuses the WordNet-backed lexical-semantic prior forest and
pushes it through a curriculum of progressively harder lexical streams, then
introduces a math-text observation stream before selectively replaying the most useful
seed nodes and finally revisiting the original seed stream:
  - in-domain Peter Rabbit + repository markdown
  - medium out-of-domain vocabulary from ``nltk.corpus.words``
  - hard out-of-domain vocabulary with a stricter lexical filter
  - a pressure shock stage with a much smaller hot cache and aggressive sweeps
  - a math-text stream generated from arithmetic observations
  - selective replay of the seed observations tied to the highest-utility seed
    learned nodes
  - the original seed batch again, after the cross-stream interference

The goal is to test whether HFN/HPM can survive a stream switch and whether
selective replay can preserve useful earlier structure better than blunt full
replay.
"""

from __future__ import annotations

import gc
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn import Observer
from hfn.tiered_forest import TieredForest
from hpm_fractal_node.math.math_loader import (
    generate_observations as generate_math_observations,
    get_category as get_math_category,
)
from hpm_fractal_node.experiments import experiment_lexical_curriculum as curriculum
from hpm_fractal_node.experiments import experiment_lexical_pressure_revisit as pressure
from hpm_fractal_node.experiments import experiment_lexical_semantic_forest as base

D = curriculum.D
SEED = curriculum.SEED
N_SAMPLES = 400
N_PASSES = 1
TAU_SIGMA = curriculum.TAU_SIGMA
TAU_MARGIN = curriculum.TAU_MARGIN

OOD_MILD_TOKEN_MIN_LEN = curriculum.OOD_MILD_TOKEN_MIN_LEN
OOD_MILD_MAX_TOKEN_POOL = curriculum.OOD_MILD_MAX_TOKEN_POOL
OOD_MILD_TOKENS_PER_OBS = curriculum.OOD_MILD_TOKENS_PER_OBS
OOD_HARD_TOKEN_MIN_LEN = curriculum.OOD_HARD_TOKEN_MIN_LEN
OOD_HARD_MAX_TOKEN_POOL = curriculum.OOD_HARD_MAX_TOKEN_POOL
OOD_HARD_TOKENS_PER_OBS = curriculum.OOD_HARD_TOKENS_PER_OBS

PRESSURE_MAX_HOT = pressure.PRESSURE_MAX_HOT
PRESSURE_SWEEP_EVERY = pressure.PRESSURE_SWEEP_EVERY
PRESSURE_MIN_FREE_RAM_MB = pressure.PRESSURE_MIN_FREE_RAM_MB
PRESSURE_TOKEN_MIN_LEN = pressure.PRESSURE_TOKEN_MIN_LEN
PRESSURE_MAX_TOKEN_POOL = pressure.PRESSURE_MAX_TOKEN_POOL
PRESSURE_TOKENS_PER_OBS = pressure.PRESSURE_TOKENS_PER_OBS

SELECT_REPLAY_NODE_LIMIT = 5
SELECT_REPLAY_MAX_OBS = 180

ObservationBatch = curriculum.ObservationBatch

NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
)
TENS_WORDS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty"}
MATH_OP_PHRASES = {
    "+": "plus",
    "-": "minus",
    "*": "times",
    "//": "floor division",
    "mod": "modulo",
    "gcd": "greatest common divisor",
    "pow": "power",
}
MATH_CATEGORY_TOKENS = {
    "identity_add": ("identity", "addition"),
    "identity_mul": ("identity", "multiplication"),
    "absorption_mul": ("absorption", "multiplication"),
    "identity_pow_zero": ("identity", "power", "zero"),
    "identity_pow_one": ("identity", "power", "one"),
    "identity_div_one": ("identity", "division", "one"),
    "exact_divisibility": ("exact", "divisibility"),
    "gcd_self": ("gcd", "self"),
    "gcd_with_zero": ("gcd", "zero"),
    "gcd_prime": ("gcd", "prime"),
    "carry": ("carry",),
    "no_carry": ("no", "carry"),
    "sub_self": ("subtract", "self"),
    "subtraction": ("subtraction",),
    "prime_result": ("prime", "result"),
    "perfect_power": ("perfect", "power"),
    "power_general": ("power", "general"),
    "mul_large": ("large", "multiplication"),
    "mul_small": ("small", "multiplication"),
    "floor_div": ("floor", "division"),
    "mod_general": ("modulo", "general"),
    "gcd_general": ("gcd", "general"),
    "general": ("general",),
}


@dataclass(frozen=True)
class StageStats:
    stage_name: str
    difficulty: str
    source_name: str
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
    hot_start: int
    max_hot: int
    sweep_every: int
    min_free_ram_mb: int


@dataclass(frozen=True)
class StageResult:
    stats: StageStats
    created_learned_ids: frozenset[str]
    explained_learned_ids: frozenset[str]
    best_ids_by_obs: tuple[str | None, ...]
    node_supports: dict[str, int]


@dataclass(frozen=True)
class PressureConfig:
    max_hot: int
    sweep_every: int
    min_free_ram_mb: int


def _number_to_words(value: int) -> str:
    if 0 <= value < len(NUMBER_WORDS):
        return NUMBER_WORDS[value]
    if 20 <= value <= 80:
        tens = (value // 10) * 10
        ones = value % 10
        if ones == 0:
            return TENS_WORDS.get(tens, str(value))
        return f"{TENS_WORDS.get(tens, str(tens))} {NUMBER_WORDS[ones]}"
    return str(value)


def _math_observation_text(left: int, op: str, right: int, result: int, category: str) -> str:
    left_words = _number_to_words(left)
    right_words = _number_to_words(right)
    result_words = _number_to_words(result)
    op_phrase = MATH_OP_PHRASES.get(op, op)
    category_tokens = " ".join(MATH_CATEGORY_TOKENS.get(category, (category.replace("_", " "),)))
    return (
        f"{left_words} {op_phrase} {right_words} equals {result_words} "
        f"math observation {category_tokens}".strip()
    )


def _build_world_model(mode: str = "large"):
    return curriculum._build_world_model(mode)


def _make_observer(forest: TieredForest, prior_ids: set[str]) -> Observer:
    return curriculum._make_observer(forest, prior_ids)


def _pressure_pool(inventory) -> tuple[str, ...]:
    return curriculum._ood_token_pool(
        inventory,
        min_len=PRESSURE_TOKEN_MIN_LEN,
        max_pool=PRESSURE_MAX_TOKEN_POOL,
        require_hard_chars=True,
    )


def _apply_pressure(forest: TieredForest) -> PressureConfig:
    baseline = PressureConfig(
        max_hot=getattr(forest, "_max_hot", 500),
        sweep_every=getattr(forest, "_sweep_every", 100),
        min_free_ram_mb=getattr(forest, "_min_free_ram_mb", 500),
    )
    forest._max_hot = min(baseline.max_hot, PRESSURE_MAX_HOT)
    forest._sweep_every = PRESSURE_SWEEP_EVERY
    forest._min_free_ram_mb = PRESSURE_MIN_FREE_RAM_MB
    try:
        forest._evict_lru_if_needed()
    except Exception:
        pass
    return baseline


def _restore_pressure(forest: TieredForest, baseline: PressureConfig) -> None:
    forest._max_hot = baseline.max_hot
    forest._sweep_every = baseline.sweep_every
    forest._min_free_ram_mb = baseline.min_free_ram_mb
    try:
        forest._evict_lru_if_needed()
    except Exception:
        pass


def _normalize_token(token: str) -> str:
    helper = getattr(base, "_normalize_token", None)
    if callable(helper):
        return helper(token)
    token = token.lower().replace("'", "")
    return "".join(ch for ch in token if ch.isalpha())


def _tokenize_text(text: str) -> list[str]:
    helper = getattr(base, "_tokenize_text", None)
    if callable(helper):
        return list(helper(text))
    tokens: list[str] = []
    for raw in text.split():
        token = _normalize_token(raw)
        if token:
            tokens.append(token)
    return tokens


def _best_corpus_anchor_with_info(inventory, tokens: list[str], seed: int):
    helper = getattr(curriculum, "_best_corpus_anchor_with_info", None)
    if callable(helper):
        try:
            return helper(inventory, tokens, seed)
        except TypeError:
            pass
    candidate_counts: dict[str, int] = defaultdict(int)
    lemma_to_synsets = getattr(inventory, "lemma_to_synsets", {})
    depth_by_synset = getattr(inventory, "depth_by_synset", {})
    for token in tokens:
        for syn_name in lemma_to_synsets.get(token, () ):  # pragma: no cover - compatibility path
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
    seeds = getattr(inventory, "seed_synsets", ())
    if not seeds:
        raise ValueError("Inventory does not expose seed_synsets for fallback anchoring")
    index = abs(hash(f"{seed}:{' '.join(tokens)}")) % len(seeds)
    syn = seeds[index]
    return inventory.synset_records[syn.name()], True


def _compose_observation_mu(anchor, text: str, relation_tag: str) -> np.ndarray:
    compose = getattr(base, "_compose_mu", None)
    if callable(compose):
        try:
            vec = compose(base._normalize_pos(anchor.synset.pos()), anchor.depth, text, relation_tag=relation_tag)
            if vec is not None:
                return vec
        except TypeError:
            pass
    return base._compose_mu(base._normalize_pos(anchor.synset.pos()), anchor.depth, text, relation_tag=relation_tag)


def _in_domain_segments() -> tuple[str, ...]:
    return curriculum._in_domain_segments()


def _build_text_observation_batch(
    inventory,
    items: list[tuple[str, str]],
    *,
    seed: int,
    source_name: str,
    relation_prefix: str,
) -> ObservationBatch:
    observations: list[tuple[np.ndarray, str, str]] = []
    known_token_hits = 0
    fallback_obs = 0
    token_total = 0
    unique_tokens: set[str] = set()

    for obs_index, (text, category) in enumerate(items):
        tokens = _tokenize_text(text)
        unique_tokens.update(tokens)
        token_total += len(tokens)
        known_token_hits += sum(1 for token in tokens if token in inventory.lemma_to_synsets)
        anchor, used_fallback = _best_corpus_anchor_with_info(inventory, tokens, seed=seed + obs_index)
        fallback_obs += int(used_fallback)
        anchor_text = f"{text} anchor {anchor.synset.name()} {anchor.label}"
        vec = _compose_observation_mu(anchor, anchor_text, relation_tag=f"{relation_prefix}:{category}:{anchor.label}")
        observations.append((vec, text, category))

    return ObservationBatch(
        observations=observations,
        known_token_hits=known_token_hits,
        fallback_obs=fallback_obs,
        token_total=token_total,
        unique_tokens=len(unique_tokens),
        source_name=source_name,
    )


def _make_math_shift_batch(inventory, n_samples: int, seed: int) -> ObservationBatch:
    sampled = generate_math_observations(n=n_samples, seed=seed)
    items: list[tuple[str, str]] = []
    for obs_index, (_, (left, op, right, result)) in enumerate(sampled):
        category = get_math_category(left, op, right, result)
        text = _math_observation_text(left, op, right, result, category)
        items.append((text, category))
    return _build_text_observation_batch(
        inventory,
        items,
        seed=seed,
        source_name="math_shift",
        relation_prefix="math",
    )


def _make_selective_replay_batch(
    inventory,
    seed_batch: ObservationBatch,
    seed_best_ids: tuple[str | None, ...],
    selected_nodes: list[str],
    *,
    seed: int,
    max_obs: int = SELECT_REPLAY_MAX_OBS,
) -> ObservationBatch:
    selected_indices = [idx for idx, node_id in enumerate(seed_best_ids) if node_id in selected_nodes]
    if not selected_indices:
        selected_indices = list(range(min(max_obs, len(seed_batch.observations))))
    if len(selected_indices) > max_obs:
        rng = np.random.default_rng(seed)
        selected_indices = list(rng.choice(selected_indices, size=max_obs, replace=False))
    observations = [seed_batch.observations[idx] for idx in selected_indices]
    items = [(text, category) for _, text, category in observations]
    return _build_text_observation_batch(
        inventory,
        items,
        seed=seed,
        source_name="selective_replay",
        relation_prefix="replay",
    )


def _select_seed_nodes(seed_result: StageResult, prior_ids: set[str], limit: int = SELECT_REPLAY_NODE_LIMIT) -> list[str]:
    ranked = [
        (node_id, count)
        for node_id, count in seed_result.node_supports.items()
        if node_id not in prior_ids
    ]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    selected = [node_id for node_id, _ in ranked[:limit]]
    if not selected:
        selected = [node_id for node_id, _ in sorted(seed_result.node_supports.items(), key=lambda item: (-item[1], item[0]))[:limit]]
    return selected


def _run_stage(
    stage_name: str,
    difficulty: str,
    batch: ObservationBatch,
    forest: TieredForest,
    prior_ids: set[str],
    observer: Observer,
) -> StageResult:
    print("\n" + "=" * 72)
    print(f"  Stage: {stage_name} ({difficulty})")
    print("=" * 72)

    start_ids = curriculum._forest_node_ids(forest)
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
    print(
        f"  Pressure settings: max_hot={getattr(forest, '_max_hot', '?')} "
        f"sweep_every={getattr(forest, '_sweep_every', '?')} "
        f"min_free_ram_mb={getattr(forest, '_min_free_ram_mb', '?')}"
    )
    print(f"  Known token rate: {batch.known_token_hits / max(batch.token_total, 1):.2%}")
    print(f"  Fallback rate: {batch.fallback_obs / max(len(batch.observations), 1):.2%}")
    print(f"  Avg tokens / obs: {batch.token_total / max(len(batch.observations), 1):.2f}")
    print(f"  Unique tokens: {batch.unique_tokens}")
    print(f"  tau = {observer.tau:.2f}")

    node_explanations: dict[str, list[str]] = defaultdict(list)
    best_ids_by_obs: list[str | None] = [None] * len(batch.observations)
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
                best_ids_by_obs[idx] = best_id
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

    end_ids = curriculum._forest_node_ids(forest)
    created_learned_ids = frozenset(node_id for node_id in (end_ids - start_ids) if node_id not in prior_ids)
    explained_learned_ids = frozenset(node_id for node_id in node_explanations if node_id not in prior_ids)
    prior_explained = sum(len(labels) for node_id, labels in node_explanations.items() if node_id in prior_ids)
    learned_explained = sum(len(node_explanations[node_id]) for node_id in explained_learned_ids)
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
    for node_id in explained_learned_ids:
        if forest.get(node_id) is None:
            continue
        labels = node_explanations[node_id]
        if len(labels) < 5:
            continue
        cat_counts: dict[str, int] = defaultdict(int)
        for category in labels:
            cat_counts[category] += 1
        cat_purities.append(base.purity(cat_counts))

    return StageResult(
        stats=StageStats(
            stage_name=stage_name,
            difficulty=difficulty,
            source_name=batch.source_name,
            wall_clock_s=elapsed,
            rss_delta_mb=(peak_rss - start_rss) / (1024**2),
            prior_nodes=len(prior_ids),
            final_node_count=len(forest),
            learned_nodes_surviving=sum(1 for node_id in end_ids if node_id not in prior_ids),
            new_learned_nodes_surviving=len(created_learned_ids),
            learned_nodes_explained=len(explained_learned_ids),
            coverage_pct=100.0 * total_attributed / max(n_obs_total, 1),
            mean_explaining_layer=(layer_rank_sum / layer_rank_n) if layer_rank_n else 0.0,
            layer_counts=dict(layer_counts),
            known_token_rate=batch.known_token_hits / max(batch.token_total, 1),
            fallback_rate=batch.fallback_obs / max(len(batch.observations), 1),
            avg_tokens=batch.token_total / max(len(batch.observations), 1),
            unique_tokens=batch.unique_tokens,
            mean_purity=float(np.mean(cat_purities)) if cat_purities else float("nan"),
            n_purity_nodes=len(cat_purities),
            hot_start=forest.hot_count(),
            max_hot=getattr(forest, "_max_hot", -1),
            sweep_every=getattr(forest, "_sweep_every", -1),
            min_free_ram_mb=getattr(forest, "_min_free_ram_mb", -1),
        ),
        created_learned_ids=created_learned_ids,
        explained_learned_ids=explained_learned_ids,
        best_ids_by_obs=tuple(best_ids_by_obs),
        node_supports={node_id: len(labels) for node_id, labels in node_explanations.items()},
    )


def _print_summary(
    results: list[StageResult],
    forest: TieredForest,
    selected_seed_nodes: list[str],
    selected_seed_remaining_after_math: int,
    selected_replay_obs_count: int,
) -> None:
    stages = [result.stats for result in results]
    result_by_name = {result.stats.stage_name: result for result in results}

    print("\n" + "=" * 72)
    print("  CROSS-STREAM SUMMARY")
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
            f"rss_delta={stage.rss_delta_mb:.1f}MB "
            f"hot_start={stage.hot_start} "
            f"max_hot={stage.max_hot} "
            f"sweep_every={stage.sweep_every} "
            f"min_free_ram_mb={stage.min_free_ram_mb} "
            f"source={stage.source_name}"
        )
        for key in ("lemma", "synset", "relation", "pos", "lexname", "depth", "ontology", "learned"):
            print(f"    layer[{key:<10}] {stage.layer_counts.get(key, 0):>10}")

    if stages:
        first = stages[0]
        last = stages[-1]
        print(f"\n  Coverage delta (last-first): {last.coverage_pct - first.coverage_pct:+.4f}%")
        print(f"  Learned nodes delta (last-first): {last.learned_nodes_surviving - first.learned_nodes_surviving:+d}")
        print(f"  Mean explaining-layer delta (last-first): {last.mean_explaining_layer - first.mean_explaining_layer:+.2f}")

    seed_result = result_by_name.get("seed")
    pressure_result = result_by_name.get("pressure_shock")
    math_result = result_by_name.get("math_shift")
    selective_result = result_by_name.get("selective_replay")
    revisit_result = result_by_name.get("revisit_seed")
    if seed_result and pressure_result and math_result and selective_result and revisit_result:
        selected_seed_set = set(selected_seed_nodes)
        reused_selected = len(selected_seed_set & selective_result.explained_learned_ids)
        print("\n  Cross-stream focus")
        print(f"  Selected seed nodes: {len(selected_seed_nodes)}")
        print(f"  Selected seed nodes remaining after math shift: {selected_seed_remaining_after_math}")
        print(f"  Selected seed nodes reused during selective replay: {reused_selected}")
        print(f"  Selective replay observations: {selected_replay_obs_count}")
        print(f"  Pressure stage learned explanations: {pressure_result.stats.learned_nodes_explained}")
        print(f"  Math stage learned explanations: {math_result.stats.learned_nodes_explained}")
        print(f"  Revisit coverage delta vs seed: {revisit_result.stats.coverage_pct - seed_result.stats.coverage_pct:+.4f}%")
        print(
            f"  Revisit mean layer delta vs seed: "
            f"{revisit_result.stats.mean_explaining_layer - seed_result.stats.mean_explaining_layer:+.2f}"
        )
        print(
            f"  Revisit learned explanations delta vs seed: "
            f"{revisit_result.stats.learned_nodes_explained - seed_result.stats.learned_nodes_explained:+d}"
        )


def main() -> None:
    print("WordNet lexical cross-stream replay experiment")
    print(f"  D={D}, N_SAMPLES={N_SAMPLES}, N_PASSES={N_PASSES}, SEED={SEED}")
    print("  Observation stream stretches a WordNet forest, switches to math-text observations, then selectively replays seed nodes.")

    mode = "compact" if N_SAMPLES <= 5 else "large"
    forest, prior_ids, inventory = _build_world_model(mode)
    segments = _in_domain_segments()
    mild_pool = curriculum._ood_token_pool(
        inventory,
        min_len=OOD_MILD_TOKEN_MIN_LEN,
        max_pool=OOD_MILD_MAX_TOKEN_POOL,
        require_hard_chars=False,
    )
    hard_pool = curriculum._ood_token_pool(
        inventory,
        min_len=OOD_HARD_TOKEN_MIN_LEN,
        max_pool=OOD_HARD_MAX_TOKEN_POOL,
        require_hard_chars=True,
    )
    pressure_pool = _pressure_pool(inventory)
    print(f"  In-domain segments available: {len(segments)}")
    print(f"  Mild OOD token pool size: {len(mild_pool)}")
    print(f"  Hard OOD token pool size: {len(hard_pool)}")
    print(f"  Pressure token pool size: {len(pressure_pool)}")
    print(f"  World-model mode: {mode}")

    observer = _make_observer(forest, prior_ids)
    baseline_pressure = PressureConfig(
        max_hot=getattr(forest, "_max_hot", 500),
        sweep_every=getattr(forest, "_sweep_every", 100),
        min_free_ram_mb=getattr(forest, "_min_free_ram_mb", 500),
    )

    seed_batch = curriculum._make_in_domain_batch(inventory, N_SAMPLES, seed=SEED)
    stretch_batch = curriculum._make_ood_batch(
        inventory,
        N_SAMPLES,
        seed=SEED + 1,
        token_pool=mild_pool,
        tokens_per_obs=OOD_MILD_TOKENS_PER_OBS,
        source_name="mild_ood",
    )
    hard_batch = curriculum._make_ood_batch(
        inventory,
        N_SAMPLES,
        seed=SEED + 2,
        token_pool=hard_pool,
        tokens_per_obs=OOD_HARD_TOKENS_PER_OBS,
        source_name="hard_ood",
    )
    pressure_batch = curriculum._make_ood_batch(
        inventory,
        N_SAMPLES,
        seed=SEED + 3,
        token_pool=pressure_pool,
        tokens_per_obs=PRESSURE_TOKENS_PER_OBS,
        source_name="pressure_ood",
    )
    math_batch = _make_math_shift_batch(inventory, N_SAMPLES, seed=SEED + 4)

    print(f"  Generated {len(seed_batch.observations)} seed observations")
    print(f"  Generated {len(stretch_batch.observations)} stretch observations")
    print(f"  Generated {len(hard_batch.observations)} hard observations")
    print(f"  Generated {len(pressure_batch.observations)} pressure observations")
    print(f"  Generated {len(math_batch.observations)} math observations")

    seed_result = _run_stage("seed", "easy", seed_batch, forest, prior_ids, observer)
    selected_seed_nodes = _select_seed_nodes(seed_result, prior_ids)
    selective_replay_batch = _make_selective_replay_batch(
        inventory,
        seed_batch,
        seed_result.best_ids_by_obs,
        selected_seed_nodes,
        seed=SEED + 5,
    )
    print(f"  Selected seed nodes for replay: {len(selected_seed_nodes)}")
    print(f"  Selective replay observations: {len(selective_replay_batch.observations)}")

    results: list[StageResult] = [seed_result]
    pressure_applied = False
    selected_seed_remaining_after_math = 0

    for stage_name, difficulty, batch, action in [
        ("stretch", "medium", stretch_batch, None),
        ("stretch_hard", "hard", hard_batch, None),
        ("pressure_shock", "shock", pressure_batch, "apply"),
        ("math_shift", "cross_stream", math_batch, None),
        ("selective_replay", "targeted_replay", selective_replay_batch, "restore"),
        ("revisit_seed", "replay", seed_batch, None),
    ]:
        if action == "apply" and not pressure_applied:
            _apply_pressure(forest)
            pressure_applied = True
        elif action == "restore" and pressure_applied:
            selected_seed_remaining_after_math = sum(1 for node_id in selected_seed_nodes if forest.get(node_id) is not None)
            _restore_pressure(forest, baseline_pressure)
            pressure_applied = False
        results.append(_run_stage(stage_name, difficulty, batch, forest, prior_ids, observer))

    if pressure_applied:
        selected_seed_remaining_after_math = sum(1 for node_id in selected_seed_nodes if forest.get(node_id) is not None)
        _restore_pressure(forest, baseline_pressure)

    _print_summary(
        results,
        forest,
        selected_seed_nodes,
        selected_seed_remaining_after_math,
        len(selective_replay_batch.observations),
    )


if __name__ == "__main__":
    main()
