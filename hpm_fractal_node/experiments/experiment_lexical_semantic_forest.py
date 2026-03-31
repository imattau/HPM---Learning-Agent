
"""
WordNet-backed large prior forest experiment.

This experiment builds a real lexical-semantic ontology from NLTK WordNet
and exercises it with a real text corpus.
Priors are layered as:
  - lemma / surface nodes
  - synset nodes
  - relation instances (hypernym / meronym / holonym closures)
  - abstraction roots (POS, lexname, depth, ontology root)

The goal is to test whether a several-thousand-node external ontology gives
HFN a more AI-like memory substrate than the synthetic toy priors.

Usage:
    PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_lexical_semantic_forest.py
"""

from __future__ import annotations

import gc
import hashlib
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

import nltk
from nltk.corpus import wordnet as wn

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn import HFN, Observer, calibrate_tau
from hfn.tiered_forest import TieredForest
from hpm.substrate.base import hash_vectorise
from hpm_fractal_node.nlp.download_corpus import corpus_path, download as download_corpus

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

D = 96
SEED = 42
N_SAMPLES = 2400
N_PASSES = 2

TAU_SIGMA = 1.0
TAU_MARGIN = 4.0

POS_ORDER = ("n", "v", "a", "r")
POS_LABEL = {"n": "noun", "v": "verb", "a": "adj", "r": "adv"}

MODE_CONFIG = {
    "compact": {
        "quotas": {"n": 200, "v": 70, "a": 40, "r": 20},
        "hypernym_depth": 1,
        "max_hot": 500,
    },
    "large": {
        "quotas": {"n": 700, "v": 200, "a": 100, "r": 50},
        "hypernym_depth": 2,
        "max_hot": 1200,
    },
}

METADATA_DIM = 16  # 4 POS + 6 depth buckets + 6 relation/meta flags
HASH_DIM = D - METADATA_DIM

DEPTH_BUCKETS = (
    (0, 0, "depth_0"),
    (1, 1, "depth_1"),
    (2, 2, "depth_2"),
    (3, 4, "depth_3_4"),
    (5, 7, "depth_5_7"),
    (8, 999, "depth_8_plus"),
)

RELATION_TYPES = (
    ("hypernym", "hypernym"),
    ("instance_hypernym", "instance_hypernym"),
    ("member_holonym", "member_holonym"),
    ("part_holonym", "part_holonym"),
)


@dataclass(frozen=True)
class SynsetRecord:
    synset: object
    label: str
    text: str
    mu: np.ndarray
    kind: str
    depth: int


@dataclass(frozen=True)
class Inventory:
    seed_synsets: list
    closure_synsets: list
    synset_records: dict[str, SynsetRecord]
    lemma_texts: dict[str, str]
    lemma_to_synsets: dict[str, list[str]]
    relation_specs: list[tuple[str, str, str]]
    depth_by_synset: dict[str, int]
    label_by_synset: dict[str, str]
    lexname_counts: dict[str, int]


def _ensure_wordnet() -> None:
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError as exc:
        raise LookupError(
            "NLTK WordNet corpus not found. Install it with: nltk.download('wordnet')"
        ) from exc


def _stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _one_hot(dim: int, idx: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float64)
    vec[idx] = 1.0
    return vec





def _normalize_pos(pos: str) -> str:
    if pos == "s":
        return "a"
    return pos


def _depth_bucket(depth: int) -> int:
    for idx, (lo, hi, _) in enumerate(DEPTH_BUCKETS):
        if lo <= depth <= hi:
            return idx
    return len(DEPTH_BUCKETS) - 1


def _metadata_vector(pos: str, depth: int, relation_tag: str = "") -> np.ndarray:
    vec = np.zeros(METADATA_DIM, dtype=np.float64)
    pos_idx = POS_ORDER.index(_normalize_pos(pos))
    vec[pos_idx] = 1.0
    bucket_idx = _depth_bucket(depth)
    vec[4 + bucket_idx] = 0.9
    if relation_tag:
        rel_idx = abs(_stable_int(relation_tag)) % 6
        vec[10 + rel_idx] = 0.7
    return vec


def _compose_mu(pos: str, depth: int, text: str, relation_tag: str = "") -> np.ndarray:
    vec = np.zeros(D, dtype=np.float64)
    vec[:METADATA_DIM] = _metadata_vector(pos, depth, relation_tag)
    vec[METADATA_DIM:] = 0.8 * hash_vectorise(text, dim=HASH_DIM)
    return vec


def _node(node_id: str, mu: np.ndarray, sigma_scale: float = 0.25) -> HFN:
    sigma = np.full(D, sigma_scale, dtype=np.float64)
    return HFN(mu=mu, sigma=sigma, id=node_id, use_diag=True)


def _synset_score(syn) -> tuple[int, int, str]:
    return (
        len(syn.lemmas()),
        len(syn.hypernyms()) + len(syn.instance_hypernyms()) + len(syn.member_holonyms()) + len(syn.part_holonyms()),
        syn.name(),
    )


def _select_synsets_for_mode(mode: str) -> list:
    quotas = MODE_CONFIG[mode]["quotas"]
    selected: list = []
    for pos in POS_ORDER:
        synsets = [s for s in wn.all_synsets(pos=pos) if s.definition()]
        synsets.sort(key=_synset_score, reverse=True)
        selected.extend(synsets[: quotas[pos]])
    return selected


def _lexname_group(lexname: str) -> str:
    parts = lexname.split(".")
    if len(parts) >= 2:
        return parts[0] + "." + parts[1]
    return lexname


def _make_synset_text(syn) -> str:
    lemmas = " ".join(l.name().replace("_", " ") for l in syn.lemmas()[:6])
    examples = " ".join(syn.examples()[:2])
    return f"{syn.name().replace('.', ' ')} {syn.definition()} {examples} {lemmas}"

def _normalize_token(token: str) -> str:
    token = token.lower().replace("'", "")
    token = re.sub(r"[^a-z]+", "", token)
    return token


def _tokenize_text(text: str) -> list[str]:
    tokens = []
    for raw in re.findall(r"[A-Za-z][A-Za-z'-]*", text):
        token = _normalize_token(raw)
        if token:
            tokens.append(token)
    return tokens


def _segment_corpus(text: str) -> list[str]:
    text = text.replace("\r", " ")
    segments = []
    for raw_segment in re.split(r"(?<=[.!?])\s+", text):
        segment = " ".join(raw_segment.split())
        if len(segment) < 24:
            continue
        segments.append(segment)
    return segments


def _load_corpus_text() -> str:
    path = corpus_path()
    if not path.exists():
        try:
            download_corpus()
        except SystemExit:
            pass
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found at {path}")
    return path.read_text(encoding="utf-8")


def _best_corpus_anchor(inventory: Inventory, tokens: list[str], seed: int) -> SynsetRecord:
    candidate_counts: dict[str, int] = defaultdict(int)
    for token in tokens:
        for syn_name in inventory.lemma_to_synsets.get(token, []):
            candidate_counts[syn_name] += 1

    if candidate_counts:
        best_name = max(
            candidate_counts,
            key=lambda name: (
                candidate_counts[name],
                -inventory.depth_by_synset.get(name, 0),
                name,
            ),
        )
        return inventory.synset_records[best_name]

    fallback = inventory.seed_synsets[_stable_int(f"{seed}:{' '.join(tokens)}") % len(inventory.seed_synsets)]
    return inventory.synset_records[fallback.name()]



def _closure_with_depth(seeds: list, max_depth: int) -> tuple[list, dict[str, int], list[tuple[str, str, str]]]:
    seen: dict[str, object] = {}
    depth_by_synset: dict[str, int] = {}
    relation_specs: list[tuple[str, str, str]] = []
    queue: list[tuple[object, int]] = [(syn, 0) for syn in seeds]

    while queue:
        syn, depth = queue.pop(0)
        name = syn.name()
        prev = depth_by_synset.get(name)
        if prev is not None and prev <= depth:
            continue
        depth_by_synset[name] = depth
        seen[name] = syn
        if depth >= max_depth:
            continue

        relation_lists = [
            ("hypernym", syn.hypernyms()),
            ("instance_hypernym", syn.instance_hypernyms()),
            ("member_holonym", syn.member_holonyms()),
            ("part_holonym", syn.part_holonyms()),
        ]
        for rel_tag, parents in relation_lists:
            for parent in parents[:2]:
                parent_name = parent.name()
                relation_specs.append((name, rel_tag, parent_name))
                next_depth = depth + 1
                prev_parent = depth_by_synset.get(parent_name)
                if prev_parent is None or next_depth < prev_parent:
                    depth_by_synset[parent_name] = next_depth
                    queue.append((parent, next_depth))

    closure = [seen[name] for name in sorted(seen)]
    return closure, depth_by_synset, relation_specs


def build_inventory(mode: str) -> Inventory:
    _ensure_wordnet()
    seeds = _select_synsets_for_mode(mode)
    max_depth = MODE_CONFIG[mode]["hypernym_depth"]
    closure_synsets, depth_by_synset, relation_specs = _closure_with_depth(seeds, max_depth=max_depth)

    synset_records: dict[str, SynsetRecord] = {}
    lemma_texts: dict[str, str] = {}
    lemma_to_synsets: dict[str, list[str]] = defaultdict(list)
    label_by_synset: dict[str, str] = {}
    lexname_counts: dict[str, int] = defaultdict(int)

    for syn in closure_synsets:
        name = syn.name()
        depth = depth_by_synset.get(name, syn.min_depth() if hasattr(syn, "min_depth") else 0)
        label = _lexname_group(syn.lexname())
        label_by_synset[name] = label
        lexname_counts[label] += 1
        text = _make_synset_text(syn)
        mu = _compose_mu(syn.pos(), depth, text)
        synset_records[name] = SynsetRecord(
            synset=syn,
            label=label,
            text=text,
            mu=mu,
            kind="synset",
            depth=depth,
        )
        for lemma in syn.lemmas():
            lemma_name = lemma.name().replace("_", " ").lower()
            if lemma_name not in lemma_texts:
                lemma_texts[lemma_name] = f"lemma {lemma_name} {label} {syn.definition()}"
            lemma_to_synsets[lemma_name].append(name)

    return Inventory(
        seed_synsets=seeds,
        closure_synsets=closure_synsets,
        synset_records=synset_records,
        lemma_texts=lemma_texts,
        lemma_to_synsets={k: sorted(set(v)) for k, v in lemma_to_synsets.items()},
        relation_specs=relation_specs,
        depth_by_synset=depth_by_synset,
        label_by_synset=label_by_synset,
        lexname_counts=dict(lexname_counts),
    )


def _build_world_model(mode: str, forest_cls=None, **tiered_kwargs) -> tuple:
    from hfn.forest import Forest as _Forest

    if forest_cls is None:
        forest_cls = TieredForest
    kwargs = tiered_kwargs if forest_cls is not _Forest else {}
    forest = forest_cls(D=D, forest_id=f"wordnet_{mode}", **kwargs)
    prior_ids: set[str] = set()
    inventory = build_inventory(mode)

    def add(node: HFN) -> HFN:
        forest.register(node)
        prior_ids.add(node.id)
        return node

    synset_nodes: dict[str, HFN] = {}
    lemma_nodes: dict[str, HFN] = {}
    relation_nodes: dict[str, HFN] = {}

    # Surface / lemma layer.
    for lemma_name, text in sorted(inventory.lemma_texts.items()):
        node_id = f"lemma_{lemma_name.replace(' ', '_')}"
        lemma_nodes[lemma_name] = add(_node(node_id, _compose_mu("n", 0, text, relation_tag="lemma"), sigma_scale=0.18))

    # Synset layer.
    for name, record in inventory.synset_records.items():
        syn = record.synset
        node = add(_node(f"syn_{name.replace('.', '_')}", record.mu, sigma_scale=0.24))
        synset_nodes[name] = node
        for lemma in syn.lemmas():
            lemma_name = lemma.name().replace("_", " ")
            lemma_node = lemma_nodes.get(lemma_name)
            if lemma_node is not None:
                node.add_child(lemma_node)

    # Relation instances.
    for src_name, rel_tag, tgt_name in inventory.relation_specs:
        src = synset_nodes.get(src_name)
        tgt = synset_nodes.get(tgt_name)
        if src is None or tgt is None:
            continue
        rel_id = f"rel_{rel_tag}_{src_name.replace('.', '_')}__{tgt_name.replace('.', '_')}"
        if rel_id in relation_nodes:
            continue
        text = f"relation {rel_tag} {src_name.replace('_', ' ')} {tgt_name.replace('_', ' ')}"
        rel_mu = _compose_mu("n", min(inventory.depth_by_synset.get(src_name, 0), inventory.depth_by_synset.get(tgt_name, 0)), text, relation_tag=rel_tag)
        rel_node = add(_node(rel_id, rel_mu, sigma_scale=0.32))
        rel_node.add_child(src)
        rel_node.add_child(tgt)
        relation_nodes[rel_id] = rel_node

    # Abstraction roots.
    pos_roots: dict[str, HFN] = {}
    pos_groups: dict[str, list[HFN]] = defaultdict(list)
    lex_groups: dict[str, list[HFN]] = defaultdict(list)
    depth_groups: dict[str, list[HFN]] = defaultdict(list)

    for name, node in synset_nodes.items():
        record = inventory.synset_records[name]
        pos_groups[record.synset.pos()].append(node)
        lex_groups[record.label].append(node)
        depth_groups[next(bucket_name for lo, hi, bucket_name in DEPTH_BUCKETS if lo <= record.depth <= hi)].append(node)

    for pos, nodes in pos_groups.items():
        pos_id = f"pos_{POS_LABEL[_normalize_pos(pos)]}"
        pos_node = add(_node(pos_id, np.mean([n.mu for n in nodes], axis=0), sigma_scale=0.4))
        for child in nodes:
            pos_node.add_child(child)
        pos_roots[pos] = pos_node

    lex_roots: dict[str, HFN] = {}
    for lexname, nodes in lex_groups.items():
        lex_id = f"lex_{lexname.replace('.', '_')}"
        lex_node = add(_node(lex_id, np.mean([n.mu for n in nodes], axis=0), sigma_scale=0.46))
        for child in nodes:
            lex_node.add_child(child)
        lex_roots[lexname] = lex_node

    depth_roots: dict[str, HFN] = {}
    for bucket_name, nodes in depth_groups.items():
        depth_node = add(_node(bucket_name, np.mean([n.mu for n in nodes], axis=0), sigma_scale=0.42))
        for child in nodes:
            depth_node.add_child(child)
        depth_roots[bucket_name] = depth_node

    ontology_root = add(_node("ontology_root", np.mean([n.mu for n in list(pos_roots.values()) + list(lex_roots.values()) + list(depth_roots.values())], axis=0), sigma_scale=0.6))
    for child in list(pos_roots.values()) + list(lex_roots.values()) + list(depth_roots.values()):
        ontology_root.add_child(child)

    return forest, prior_ids, inventory, {
        "synset_nodes": synset_nodes,
        "lemma_nodes": lemma_nodes,
        "relation_nodes": relation_nodes,
        "pos_roots": pos_roots,
        "lex_roots": lex_roots,
        "depth_roots": depth_roots,
        "ontology_root": ontology_root,
    }


def _observation_vector(syn) -> np.ndarray:
    text = _make_synset_text(syn)
    vec = np.zeros(D, dtype=np.float64)
    vec[:METADATA_DIM] = _metadata_vector(_normalize_pos(syn.pos()), syn.min_depth() if hasattr(syn, "min_depth") else 0, relation_tag=syn.lexname())
    vec[METADATA_DIM:] = 0.8 * hash_vectorise(text, dim=HASH_DIM)
    return vec


def generate_observations(inventory: Inventory, n_samples: int, seed: int) -> list[tuple[np.ndarray, str, str]]:
    corpus_text = _load_corpus_text()
    segments = _segment_corpus(corpus_text)
    if not segments:
        raise ValueError("No corpus segments were found in the external text corpus")

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(segments), size=n_samples, replace=True)
    data: list[tuple[np.ndarray, str, str]] = []
    for obs_index, segment_idx in enumerate(idxs):
        sentence = segments[segment_idx]
        tokens = _tokenize_text(sentence)
        anchor = _best_corpus_anchor(inventory, tokens, seed=seed + obs_index)
        anchor_text = f"{sentence} anchor {anchor.synset.name()} {anchor.label}"
        vec = _compose_mu(
            _normalize_pos(anchor.synset.pos()),
            anchor.depth,
            anchor_text,
            relation_tag=f"corpus:{anchor.label}",
        )
        data.append((vec, sentence, anchor.label))
    return data


def purity(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(counts.values()) / total


def _layer_of(node_id: str) -> str:
    if node_id.startswith("lemma_"):
        return "lemma"
    if node_id.startswith("syn_"):
        return "synset"
    if node_id.startswith("rel_"):
        return "relation"
    if node_id.startswith("pos_"):
        return "pos"
    if node_id.startswith("lex_"):
        return "lexname"
    if node_id.startswith("depth_"):
        return "depth"
    if node_id == "ontology_root":
        return "ontology"
    return "learned"


def _layer_rank(layer: str) -> int:
    return {
        "lemma": 0,
        "synset": 1,
        "relation": 2,
        "pos": 3,
        "lexname": 3,
        "depth": 3,
        "ontology": 4,
        "learned": 5,
    }.get(layer, 5)


def run_mode(mode: str, data: list[tuple[np.ndarray, str, str]]) -> dict:
    print(f"\n{'=' * 72}")
    print(f"  Mode: {mode}")
    print(f"{'=' * 72}")

    cold_dir = Path(__file__).parents[2] / "data" / f"hfn_wordnet_cold_{mode}"
    cold_dir.mkdir(parents=True, exist_ok=True)
    forest, prior_ids, inventory, components = _build_world_model(
        mode=mode,
        forest_cls=TieredForest,
        cold_dir=cold_dir,
        max_hot=MODE_CONFIG[mode]["max_hot"],
        sweep_every=150,
        min_free_ram_mb=256,
    )
    forest.set_protected(prior_ids)

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
        layer = _layer_of(node_id)
        layer_counts[layer] += len(labels)
        layer_rank_sum += _layer_rank(layer) * len(labels)
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
        cat_purities.append(purity(cat_counts))

    mean_purity = float(np.mean(cat_purities)) if cat_purities else float("nan")
    mean_layer = layer_rank_sum / layer_rank_n if layer_rank_n else 0.0

    return {
        "mode": mode,
        "elapsed_s": elapsed,
        "peak_rss_delta_mb": (peak_rss - start_rss) / (1024 ** 2),
        "n_priors": len(prior_ids),
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
    }


def print_comparison(compact: dict, large: dict) -> None:
    print(f"\n{'=' * 72}")
    print("  COMPARISON")
    print(f"{'=' * 72}")
    rows = [
        ("Wall-clock time (s)", f"{compact['elapsed_s']:.2f}", f"{large['elapsed_s']:.2f}"),
        ("Peak RSS delta (MB)", f"{compact['peak_rss_delta_mb']:.1f}", f"{large['peak_rss_delta_mb']:.1f}"),
        ("Prior nodes", f"{compact['n_priors']}", f"{large['n_priors']}"),
        ("Final node count", f"{compact['final_node_count']}", f"{large['final_node_count']}"),
        ("Learned nodes surviving", f"{compact['learned_nodes_surviving']}", f"{large['learned_nodes_surviving']}"),
        ("Learned nodes explained", f"{compact['learned_nodes_explained']}", f"{large['learned_nodes_explained']}"),
        ("Coverage %", f"{compact['coverage_pct']:.2f}%", f"{large['coverage_pct']:.2f}%"),
        ("Mean category purity (n>=5)", f"{compact['mean_purity']:.3f}", f"{large['mean_purity']:.3f}"),
        ("Purity-eligible nodes", f"{compact['n_purity_nodes']}", f"{large['n_purity_nodes']}"),
        ("Mean explaining layer", f"{compact['mean_explaining_layer']:.2f}", f"{large['mean_explaining_layer']:.2f}"),
    ]
    print(f"  {'Metric':<30} {'Compact':>14} {'Large':>14}")
    print("  " + "-" * 60)
    for metric, left, right in rows:
        print(f"  {metric:<30} {left:>14} {right:>14}")

    for key in ("lemma", "synset", "relation", "pos", "lexname", "depth", "ontology", "learned"):
        c = compact["layer_counts"].get(key, 0)
        l = large["layer_counts"].get(key, 0)
        print(f"  layer[{key:<10}] {c:>10} {l:>10}")

    if compact["coverage_pct"] > 0:
        print(f"\n  Coverage delta: {large['coverage_pct'] - compact['coverage_pct']:+.4f}%")
    if compact["peak_rss_delta_mb"] > 0:
        saving = (1.0 - large["peak_rss_delta_mb"] / max(compact["peak_rss_delta_mb"], 1e-9)) * 100.0
        print(f"  Peak RSS saving: {saving:+.1f}%")
def main() -> None:
    print("WordNet-backed large-prior experiment")
    print(f"  D={D}, N_SAMPLES={N_SAMPLES}, N_PASSES={N_PASSES}, SEED={SEED}")
    print("  Observation stream is sampled from Peter Rabbit corpus text and grounded in the large-mode WordNet ontology.")

    large_inventory = build_inventory("large")
    data = generate_observations(large_inventory, N_SAMPLES, seed=SEED)
    print(f"  Generated {len(data)} corpus observations from {corpus_path().name}")

    compact = run_mode("compact", data)
    large = run_mode("large", data)
    print_comparison(compact, large)


if __name__ == "__main__":
    main()
