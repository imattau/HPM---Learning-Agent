"""ARC-AGI-2 structured-agent comparison benchmark.

This benchmark keeps the structured-agent stack at L1-L5:
- L1-L3: structured ARC encoders and per-level agents
- L4: ridge head that predicts L3 from L2
- L5: metacognitive monitor that gates the L4/L3 blend

The comparison is baseline structured agents vs completion-enhanced structured
agents, where the completion path enables relational/message/identity context
and field constraints.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.arc_encoders import ArcL1Encoder, ArcL2Encoder, ArcL3Encoder
from benchmarks.common import print_results_table
from benchmarks.multi_agent_common import make_orchestrator
from hpm.agents.hierarchical import extract_relational_bundle
from hpm.agents.l4_generative import L4GenerativeHead
from hpm.agents.l5_monitor import L5MetaMonitor
from hpm.agents.structured import StructuredOrchestrator
from hpm.completion_types import FieldConstraint
from hpm.decomposition import ArcDecompositionProfile, build_arc_decomposition_profile, extract_arc_decomposition, score_arc_decomposition
from hpm.promotion import PromotionLedger, PromotionRule, best_promotable_pattern

MAX_GRID_DIM = 30
GRID_SIZE = MAX_GRID_DIM * MAX_GRID_DIM
TRAIN_REPS = 10
N_DISTRACTORS = 4
ARC_2_ROOT_ENV = "ARC_AGI_2_ROOT"

# Fixed projection matrix so benchmark comparisons stay deterministic.
_PROJ = np.random.default_rng(0).standard_normal((GRID_SIZE, 64)) / np.sqrt(64)

ARC_AGENT_OVERRIDES = dict(
    evaluator_arbitration_mode="adaptive",
    meta_evaluator_learning_rate=0.2,
    lifecycle_decay_rate=0.08,
    lifecycle_consolidation_window=2,
    lifecycle_absence_window=2,
    lifecycle_stable_weight_threshold=0.2,
    lifecycle_retire_weight_threshold=0.04,
)

ARC_ABLATION_CONDITIONS = (
    "baseline",
    "completion",
    "completion_no_identity",
    "completion_no_constraints",
    "completion_no_meta_eval",
)


def _encode_grid(grid: list[list[int]]) -> np.ndarray:
    flat: list[float] = []
    for row in grid:
        for val in row:
            flat.append(float(val) / 9.0)
    flat.extend([0.0] * (GRID_SIZE - len(flat)))
    return np.array(flat[:GRID_SIZE], dtype=float)


def encode_pair(input_grid: list[list[int]], output_grid: list[list[int]]) -> np.ndarray:
    return (_encode_grid(output_grid) - _encode_grid(input_grid)) @ _PROJ


def grid_fits(grid: list[list[int]]) -> bool:
    if len(grid) > MAX_GRID_DIM:
        return False
    return all(len(row) <= MAX_GRID_DIM for row in grid)


def task_fits(task: dict) -> bool:
    for pair in task["train"]:
        if not grid_fits(pair["input"]) or not grid_fits(pair["output"]):
            return False
    for pair in task["test"]:
        if not grid_fits(pair["input"]) or not grid_fits(pair["output"]):
            return False
    return True


def _load_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_tasks(root: str | Path | None = None, split: str = "evaluation") -> list[dict]:
    """Load ARC-AGI-2 tasks from a local checkout of the official dataset repo."""
    base = Path(root or os.environ.get(ARC_2_ROOT_ENV, "data/ARC-AGI-2")).expanduser()
    if (base / split).is_dir():
        data_dir = base / split
    elif (base / "data" / split).is_dir():
        data_dir = base / "data" / split
    else:
        raise FileNotFoundError(
            f"Could not find ARC-AGI-2 split '{split}' under {base}. "
            f"Set {ARC_2_ROOT_ENV} to the ARC-AGI-2 repo root or data directory."
        )
    return [_load_json_file(path) for path in sorted(data_dir.glob("*.json"))]


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(1.0 - cos_sim, 0.0, 2.0))


class CompletionArcL2(ArcL2Encoder):
    def encode(self, observation, epistemic=None, relational_bundles=None, structural_messages=None, identity_snapshots=None):
        vecs = super().encode(observation, epistemic=epistemic)
        if not vecs:
            return vecs
        rel = 0.0
        if relational_bundles:
            confs = [edge.confidence for bundle in relational_bundles for edge in getattr(bundle, "relations", ())]
            if confs:
                rel = float(np.mean(confs))
        msg = 0.0
        if structural_messages:
            confs = [getattr(message, "confidence", 0.0) for _, message in structural_messages]
            if confs:
                msg = float(np.mean(confs))
        lineage = 0.0
        if identity_snapshots:
            total = 0
            stable = 0
            for snapshot in identity_snapshots:
                total += len(snapshot)
                for entry in snapshot.values():
                    if entry.get("state", {}).get("lifecycle_state") == "stable":
                        stable += 1
            if total:
                lineage = stable / total
        out = []
        for vec in vecs:
            v = vec.copy()
            v[7] = float(np.clip(v[7] + 0.15 * rel, 0.0, 1.0))
            v[8] = float(np.clip(v[8] + 0.15 * msg + 0.10 * lineage, 0.0, 1.0))
            out.append(v)
        return out


class CompletionArcL3(ArcL3Encoder):
    def encode(self, observation, epistemic=None, relational_bundles=None, structural_messages=None, identity_snapshots=None):
        vecs = super().encode(observation, epistemic=epistemic)
        if not vecs:
            return vecs
        rel = 0.0
        if relational_bundles:
            confs = [edge.confidence for bundle in relational_bundles for edge in getattr(bundle, "relations", ())]
            if confs:
                rel = float(np.mean(confs))
        msg = 0.0
        if structural_messages:
            confs = [getattr(message, "confidence", 0.0) for _, message in structural_messages]
            if confs:
                msg = float(np.mean(confs))
        lineage = 0.0
        if identity_snapshots:
            total = 0
            stable = 0
            seen = 0.0
            for snapshot in identity_snapshots:
                total += len(snapshot)
                for entry in snapshot.values():
                    if entry.get("state", {}).get("lifecycle_state") == "stable":
                        stable += 1
                    seen += float(entry.get("identity", {}).get("last_seen_at", 0))
            if total:
                lineage = float(np.clip((stable + 0.05 * seen) / total, 0.0, 1.0))
        out = []
        for vec in vecs:
            v = vec.copy()
            v[12] = float(np.clip(v[12] + 0.20 * rel + 0.10 * lineage, 0.0, 1.0))
            v[13] = float(np.clip(v[13] + 0.20 * msg + 0.10 * lineage, 0.0, 1.0))
            out.append(v)
        return out


def _mean_vecs(vecs: list[np.ndarray], dim: int) -> np.ndarray:
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim, dtype=float)


def _context_from_agents(agents, include_identity_snapshots: bool = True) -> tuple[list, list, list]:
    bundles = [extract_relational_bundle(agent) for agent in agents]
    messages = []
    snapshots = []
    for agent in agents:
        if hasattr(agent, "consume_structural_inbox"):
            messages.extend(agent.consume_structural_inbox(clear=False))
        if include_identity_snapshots:
            lifecycle = getattr(agent, "_lifecycle", None)
            if lifecycle is not None and hasattr(lifecycle, "snapshot"):
                snapshots.append(lifecycle.snapshot())
            else:
                snapshots.append({})
        else:
            snapshots.append({})
    return bundles, messages, snapshots


def _grid_complexity(candidate: list[list[int]]) -> float:
    arr = np.asarray(candidate, dtype=float)
    if arr.size == 0:
        return 0.0
    if arr.ndim != 2:
        arr = np.atleast_2d(arr)
    rows, cols = arr.shape
    nonzero = float(np.count_nonzero(arr)) / float(arr.size)
    colors = {int(v) for v in arr.flat if int(v) != 0}
    color_diversity = min(len(colors) / 8.0, 1.0)
    horiz = float(np.mean(arr[:, 1:] != arr[:, :-1])) if cols > 1 else 0.0
    vert = float(np.mean(arr[1:, :] != arr[:-1, :])) if rows > 1 else 0.0
    symmetry = 0.0
    symmetry_count = 0
    if cols > 1:
        symmetry += 1.0 - float(np.mean(arr != arr[:, ::-1]))
        symmetry_count += 1
    if rows > 1:
        symmetry += 1.0 - float(np.mean(arr != arr[::-1, :]))
        symmetry_count += 1
    if symmetry_count:
        symmetry /= float(symmetry_count)
    transitions = 0.5 * (horiz + vert)
    return float(np.clip(0.35 * nonzero + 0.25 * color_diversity + 0.25 * transitions + 0.15 * (1.0 - symmetry), 0.0, 1.0))


def _identity_trust(identity_snapshots: list[dict]) -> float:
    total = 0
    stable = 0
    aged = 0.0
    for snapshot in identity_snapshots:
        for entry in snapshot.values():
            total += 1
            state = entry.get("state", {})
            identity = entry.get("identity", {})
            if state.get("lifecycle_state") == "stable":
                stable += 1
            aged += float(identity.get("last_seen_at", 0))
    if total <= 0:
        return 0.0
    stability = stable / total
    recency = float(np.tanh(aged / max(1.0, 10.0 * total)))
    return float(np.clip(0.75 * stability + 0.25 * recency, 0.0, 1.0))


def _field_constraint_adjustment(candidate: list[list[int]], constraints) -> float:
    if not constraints:
        return 0.0
    complexity = _grid_complexity(candidate)
    simple_bias = 1.0 - complexity
    adjustment = 0.0
    for constraint in constraints:
        strength = float(np.clip(constraint.strength, 0.0, 1.0))
        ctype = constraint.constraint_type
        if ctype == "penalize_complexity":
            adjustment += 0.35 * strength * complexity
        elif ctype == "prefer_simple":
            adjustment -= 0.35 * strength * simple_bias
        elif ctype == "prefer_high_level":
            adjustment -= 0.20 * strength * complexity
        elif ctype == "prefer_low_level":
            adjustment += 0.20 * strength * complexity
        else:
            adjustment += 0.05 * strength * complexity
    return float(adjustment)


def _grid_signature(grid: list[list[int]]) -> dict[str, float | tuple[int, int]]:
    arr = np.asarray(grid, dtype=int)
    if arr.size == 0:
        return {
            "shape": (0, 0),
            "colors": 0,
            "nonzero": 0.0,
            "symmetry_h": 0.0,
            "symmetry_v": 0.0,
            "transitions": 0.0,
            "center_mass": 0.0,
        }
    if arr.ndim != 2:
        arr = np.atleast_2d(arr)
    rows, cols = arr.shape
    colors = {int(v) for v in arr.flat if int(v) != 0}
    nonzero = float(np.count_nonzero(arr)) / float(arr.size)
    symmetry_h = 1.0 - float(np.mean(arr != arr[:, ::-1])) if cols > 1 else 0.0
    symmetry_v = 1.0 - float(np.mean(arr != arr[::-1, :])) if rows > 1 else 0.0
    transitions = 0.5 * (
        float(np.mean(arr[:, 1:] != arr[:, :-1])) if cols > 1 else 0.0
        + float(np.mean(arr[1:, :] != arr[:-1, :])) if rows > 1 else 0.0
    )
    r_idx, c_idx = np.indices(arr.shape)
    mass = float(arr.sum()) if float(arr.sum()) != 0.0 else 1.0
    center_mass = float((r_idx * arr).sum() + (c_idx * arr).sum()) / (2.0 * mass)
    return {
        "shape": (rows, cols),
        "colors": float(len(colors)),
        "nonzero": nonzero,
        "symmetry_h": symmetry_h,
        "symmetry_v": symmetry_v,
        "transitions": transitions,
        "center_mass": center_mass,
    }


def _signature_delta(input_sig: dict[str, float | tuple[int, int]], output_sig: dict[str, float | tuple[int, int]]) -> dict[str, float]:
    return {
        "shape_delta": abs(output_sig["shape"][0] - input_sig["shape"][0]) + abs(output_sig["shape"][1] - input_sig["shape"][1]),
        "color_delta": abs(output_sig["colors"] - input_sig["colors"]),
        "symmetry_delta": (output_sig["symmetry_h"] + output_sig["symmetry_v"]) - (input_sig["symmetry_h"] + input_sig["symmetry_v"]),
        "transition_delta": output_sig["transitions"] - input_sig["transitions"],
        "center_delta": abs(output_sig["center_mass"] - input_sig["center_mass"]),
    }


@dataclass(frozen=True)
class ArcHypothesis:
    name: str
    strength: float
    features: dict[str, float]


def _arc_hypotheses_from_train(train_pairs: list[dict]) -> list[ArcHypothesis]:
    if not train_pairs:
        return [ArcHypothesis("identity", 1.0, {"kind": "fallback"})]

    deltas = []
    for pair in train_pairs:
        deltas.append(_signature_delta(_grid_signature(pair["input"]), _grid_signature(pair["output"])))

    mean_shape = float(np.mean([d["shape_delta"] for d in deltas]))
    mean_color = float(np.mean([d["color_delta"] for d in deltas]))
    mean_sym = float(np.mean([d["symmetry_delta"] for d in deltas]))
    mean_trans = float(np.mean([d["transition_delta"] for d in deltas]))
    mean_center = float(np.mean([d["center_delta"] for d in deltas]))

    hypotheses = [
        ArcHypothesis("identity", max(0.0, 1.0 - (mean_shape + mean_color + mean_center) / 3.0), {"shape_delta": mean_shape, "color_delta": mean_color, "transition_delta": mean_trans}),
        ArcHypothesis("color_change", max(0.0, mean_color + 0.5 * mean_sym), {"color_delta": mean_color, "symmetry_delta": mean_sym}),
        ArcHypothesis("symmetry_preserve", max(0.0, 1.0 - abs(mean_sym)), {"symmetry_delta": mean_sym, "transition_delta": mean_trans}),
        ArcHypothesis("translation", max(0.0, 1.0 - min(mean_shape + mean_center, 1.0)), {"shape_delta": mean_shape, "center_delta": mean_center}),
        ArcHypothesis("structure_transform", max(0.0, mean_trans + 0.5 * mean_shape), {"transition_delta": mean_trans, "shape_delta": mean_shape}),
    ]
    hypotheses.sort(key=lambda hyp: (-hyp.strength, hyp.name))
    return hypotheses


def _hypothesis_alignment(candidate: list[list[int]], test_input: list[list[int]], hypotheses: list[ArcHypothesis]) -> tuple[float, str, float]:
    if not hypotheses:
        return 0.0, "fallback", 0.0
    cand_delta = _signature_delta(_grid_signature(test_input), _grid_signature(candidate))
    scores = []
    for hyp in hypotheses:
        h = hyp.features
        score = 0.0
        for key in ("shape_delta", "color_delta", "symmetry_delta", "transition_delta", "center_delta"):
            target = float(h.get(key, 0.0))
            actual = float(cand_delta.get(key, 0.0))
            scale = max(abs(target), abs(actual), 1.0)
            score += abs(target - actual) / scale
        scores.append((score, hyp.name, hyp.strength))
    scores.sort(key=lambda item: (item[0], -item[2], item[1]))
    best_score, best_name, best_strength = scores[0]
    return float(best_score), best_name, float(best_strength)


def _apply_rank_margin_reweight(candidate_rows: list[dict[str, float]], completion: bool) -> list[float]:
    if not completion or len(candidate_rows) <= 1:
        return [row["base"] for row in candidate_rows]

    adjustments, _ = _compute_rank_margin_adjustments(candidate_rows, include_contributions=False)
    return [float(row["base"] - adj) for row, adj in zip(candidate_rows, adjustments)]


def _apply_rank_margin_reweight_with_details(candidate_rows: list[dict[str, float]]) -> tuple[list[float], list[dict[str, float]]]:
    adjustments, contributions = _compute_rank_margin_adjustments(candidate_rows, include_contributions=True)
    scores = [float(row["base"] - adj) for row, adj in zip(candidate_rows, adjustments)]
    return scores, contributions


def _compute_rank_margin_adjustments(candidate_rows: list[dict[str, float]], include_contributions: bool = False) -> tuple[np.ndarray, list[dict[str, float]]]:
    feature_names = ("l1", "l2", "l3", "core")
    feature_matrix = np.array([[row[name] for name in feature_names] for row in candidate_rows], dtype=float)
    n_candidates = feature_matrix.shape[0]
    if n_candidates <= 0:
        return np.zeros(0, dtype=float), []

    ranks = np.argsort(np.argsort(feature_matrix, axis=0), axis=0).astype(float)
    norm_ranks = ranks / float(max(1, n_candidates - 1))
    spreads = np.maximum(np.std(feature_matrix, axis=0), 1e-6)
    sorted_scores = np.sort(feature_matrix, axis=0)
    margin_strength = np.clip((sorted_scores[1] - feature_matrix) / spreads, 0.0, 3.0) if n_candidates > 1 else np.zeros_like(feature_matrix)

    consensus = 1.0 - np.mean(norm_ranks, axis=1)
    alignment = 1.0 - np.std(norm_ranks, axis=1)
    gap_bonus = np.mean(margin_strength, axis=1)
    trust = np.array([
        0.6 * row.get("identity_trust", 0.0) + 0.4 * row.get("meta_trust", 0.0)
        for row in candidate_rows
    ], dtype=float)
    structure = np.array([row.get("structure_trust", 0.0) for row in candidate_rows], dtype=float)
    complexity = np.array([row.get("candidate_complexity", 0.0) for row in candidate_rows], dtype=float)
    hypothesis_alignment = np.array([row.get("hypothesis_alignment", 0.0) for row in candidate_rows], dtype=float)
    hypothesis_strength = np.array([row.get("hypothesis_strength", 0.0) for row in candidate_rows], dtype=float)
    part_alignment = np.array([row.get("part_alignment", 0.0) for row in candidate_rows], dtype=float)
    part_support = np.array([row.get("part_support", 0.0) for row in candidate_rows], dtype=float)
    part_ambiguity = np.array([row.get("part_ambiguity", 0.0) for row in candidate_rows], dtype=float)
    part_selected_ratio = np.array([row.get("part_selected_ratio", 0.0) for row in candidate_rows], dtype=float)
    promotion_bonus = np.array([row.get("promotion_bonus", 0.0) for row in candidate_rows], dtype=float)
    promotion_reused = np.array([row.get("promotion_reused", 0.0) for row in candidate_rows], dtype=float)

    adjustment = (
        0.14 * consensus
        + 0.10 * alignment
        + 0.08 * trust
        + 0.06 * structure
        + 0.12 * gap_bonus
        + 0.08 * hypothesis_strength
        + 0.06 * part_support
        + 0.04 * part_selected_ratio
        + 0.14 * promotion_bonus
        + 0.03 * promotion_reused
        - 0.10 * part_alignment
        - 0.04 * part_ambiguity
        - 0.06 * hypothesis_alignment
        - 0.04 * complexity
    )

    if include_contributions:
        contributions = []
        for idx, row in enumerate(candidate_rows):
            contributions.append({
                "consensus": float(0.14 * consensus[idx]),
                "alignment": float(0.10 * alignment[idx]),
                "trust": float(0.08 * trust[idx]),
                "structure": float(0.06 * structure[idx]),
                "gap_bonus": float(0.12 * gap_bonus[idx]),
                "hypothesis_strength": float(0.08 * hypothesis_strength[idx]),
                "part_support": float(0.06 * part_support[idx]),
                "part_selected_ratio": float(0.04 * part_selected_ratio[idx]),
                "promotion_bonus": float(0.14 * promotion_bonus[idx]),
                "promotion_reused": float(0.03 * promotion_reused[idx]),
                "part_alignment": float(-0.10 * part_alignment[idx]),
                "part_ambiguity": float(-0.04 * part_ambiguity[idx]),
                "hypothesis_alignment": float(-0.06 * hypothesis_alignment[idx]),
                "complexity": float(-0.04 * complexity[idx]),
                "adjustment": float(adjustment[idx]),
                "base": float(row.get("base", 0.0)),
                "score": float(row.get("base", 0.0) - adjustment[idx]),
            })
    else:
        contributions = []
    return adjustment, contributions
def _encode_with_optional_context(encoder, observation, epistemic, relational_bundles=None, structural_messages=None, identity_snapshots=None):
    params = inspect.signature(encoder.encode).parameters
    kwargs = {"epistemic": epistemic}
    if "relational_bundles" in params:
        kwargs["relational_bundles"] = relational_bundles
    if "structural_messages" in params:
        kwargs["structural_messages"] = structural_messages
    if "identity_snapshots" in params:
        kwargs["identity_snapshots"] = identity_snapshots
    return encoder.encode(observation, **kwargs)


def _make_level_orchestrator(
    feature_dim: int,
    agent_ids: list[str],
    seed: int,
    pattern_types: list[str] | None = None,
    completion: bool = False,
    adaptive_meta_eval: bool = True,
):
    agent_kwargs = dict(
        seed=seed,
        T_recomb=1,
        recomb_cooldown=1,
        min_recomb_level=1,
        kappa_max=1.0,
        N_recomb=1,
        conflict_threshold=2.0,
    )
    if completion and adaptive_meta_eval:
        agent_kwargs.update(ARC_AGENT_OVERRIDES)
    pattern_types = pattern_types or ["gaussian"] * len(agent_ids)
    return make_orchestrator(
        n_agents=len(agent_ids),
        feature_dim=feature_dim,
        agent_ids=agent_ids,
        with_monitor=False,
        gamma_soc=0.5,
        init_sigma=2.0,
        pattern_types=pattern_types,
        agent_seeds=[seed + 101 * (i + 1) for i in range(len(agent_ids))],
        **agent_kwargs,
    )


def _make_structured_stack(
    seed: int,
    completion: bool = False,
    include_constraints: bool = True,
    include_identity_snapshots: bool = True,
    adaptive_meta_eval: bool = True,
):
    l1_orch, l1_agents, _ = _make_level_orchestrator(
        64,
        ["arc_l1_0", "arc_l1_1"],
        seed + 1,
        pattern_types=["gaussian", "laplace"],
        completion=completion,
        adaptive_meta_eval=adaptive_meta_eval,
    )
    l2_orch, l2_agents, _ = _make_level_orchestrator(
        9,
        ["arc_l2_0", "arc_l2_1"],
        seed + 11,
        pattern_types=["laplace", "gaussian"],
        completion=completion,
        adaptive_meta_eval=adaptive_meta_eval,
    )
    l3_orch, l3_agents, _ = _make_level_orchestrator(
        14,
        ["arc_l3_0"],
        seed + 21,
        pattern_types=["gaussian"],
        completion=completion,
        adaptive_meta_eval=adaptive_meta_eval,
    )

    if completion and include_constraints:
        for orch in (l1_orch, l2_orch, l3_orch):
            if orch.field is not None:
                orch.field.add_constraint(FieldConstraint("prefer_simple", "*", 0.6, "benchmark", seed))
                orch.field.add_constraint(FieldConstraint("penalize_complexity", "*", 0.4, "benchmark", seed))

    enc1 = ArcL1Encoder()
    enc2 = CompletionArcL2() if completion else ArcL2Encoder()
    enc3 = CompletionArcL3() if completion else ArcL3Encoder()

    orch = StructuredOrchestrator(
        encoders=[enc1, enc2, enc3],
        orches=[l1_orch, l2_orch, l3_orch],
        agents=[l1_agents, l2_agents, l3_agents],
        level_Ks=[1, 1, 3],
        relational_bundles_enabled=completion,
        structural_messages_to_encoders_enabled=completion,
        identity_snapshots_to_encoders_enabled=completion and include_identity_snapshots,
    )
    return orch, (l1_agents, l2_agents, l3_agents), (enc1, enc2, enc3)

def _condition_settings(condition: str) -> dict[str, bool]:
    if condition == "baseline":
        return dict(completion=False, include_constraints=False, include_identity_snapshots=False, adaptive_meta_eval=False)
    if condition in {"completion", "completion_full"}:
        return dict(completion=True, include_constraints=True, include_identity_snapshots=True, adaptive_meta_eval=True)
    if condition == "completion_no_identity":
        return dict(completion=True, include_constraints=True, include_identity_snapshots=False, adaptive_meta_eval=True)
    if condition == "completion_no_constraints":
        return dict(completion=True, include_constraints=False, include_identity_snapshots=True, adaptive_meta_eval=True)
    if condition == "completion_no_meta_eval":
        return dict(completion=True, include_constraints=True, include_identity_snapshots=True, adaptive_meta_eval=False)
    raise ValueError(f"Unknown ARC-AGI-2 condition: {condition}")


def _candidate_bundle(task: dict, all_tasks: list[dict], task_idx: int):
    if "candidates" in task and "test_output" in task and "test_input" in task:
        return list(task["candidates"]), task["test_input"], task["test_output"]

    test_pair = task["test"][0]
    if "output" not in test_pair:
        raise ValueError("ARC-AGI-2 task does not provide a test output or candidates")

    test_input = test_pair["input"]
    correct_output = test_pair["output"]
    rng = np.random.default_rng(task_idx)
    other_indices = [j for j in range(len(all_tasks)) if j != task_idx and all_tasks[j].get("test") and "output" in all_tasks[j]["test"][0]]
    if len(other_indices) < N_DISTRACTORS:
        raise ValueError("Not enough eligible distractor tasks with test outputs")
    distractor_indices = rng.choice(other_indices, size=N_DISTRACTORS, replace=False)
    distractors = [all_tasks[j]["test"][0]["output"] for j in distractor_indices]
    return [correct_output, *distractors], test_input, correct_output


def _candidate_feature_bundle(orch, agents, encoders, test_input, candidate, completion: bool, include_identity_snapshots: bool, field_constraints, hypotheses, l1_proto, l2_proto, l3_proto) -> dict[str, float]:
    (l1_agents, l2_agents, l3_agents) = agents
    (enc1, enc2, enc3) = encoders
    ep1 = orch._epistemic[0] if len(orch._epistemic) > 0 else None
    ep2 = orch._epistemic[1] if len(orch._epistemic) > 1 else None

    l2_ctx = (
        _context_from_agents(l1_agents, include_identity_snapshots=include_identity_snapshots)
        if completion
        else (None, None, None)
    )
    l3_ctx = (
        _context_from_agents(l2_agents, include_identity_snapshots=include_identity_snapshots)
        if completion
        else (None, None, None)
    )

    identity_trust = _identity_trust(l2_ctx[2] + l3_ctx[2]) if completion and include_identity_snapshots else 0.0
    candidate_complexity = _grid_complexity(candidate)
    constraint_adjustment = _field_constraint_adjustment(candidate, field_constraints) if completion else 0.0
    hypothesis_alignment, hypothesis_name, hypothesis_strength = _hypothesis_alignment(candidate, test_input, hypotheses)

    l1_vec = _mean_vecs(_encode_with_optional_context(enc1, (test_input, candidate), None), 64)
    l2_vec = _mean_vecs(_encode_with_optional_context(enc2, (test_input, candidate), ep1, *l2_ctx), 9)
    l3_vec = _mean_vecs(_encode_with_optional_context(enc3, (test_input, candidate), ep2, *l3_ctx), 14)

    l1_score = float(np.sum((l1_vec - l1_proto) ** 2))
    l2_score = float(np.sum((l2_vec - l2_proto) ** 2))
    l3_score = float(np.sum((l3_vec - l3_proto) ** 2))

    if not completion:
        return {
            "base": l1_score + l2_score + l3_score,
            "l1": l1_score,
            "l2": l2_score,
            "l3": l3_score,
            "core": l3_score,
            "identity_trust": 0.0,
            "constraint_adjustment": 0.0,
            "candidate_complexity": candidate_complexity,
            "meta_trust": 0.0,
            "structure_trust": 0.0,
            "hypothesis_alignment": hypothesis_alignment,
            "hypothesis_strength": 1.0,
            "hypothesis_name": hypothesis_name,
        }

    head = orch._completion_head
    monitor = orch._completion_monitor
    l4_pred = head.predict(l2_vec)
    if l4_pred is None:
        core = l3_score
        meta_trust = 0.0
        structure_trust = 0.0
    else:
        l4_score = _cosine_distance(l4_pred, l3_vec)
        gamma = monitor.strategic_confidence() if monitor is not None else 1.0
        structure_trust = 0.45 + 0.55 * identity_trust
        meta_trust = 0.55 + 0.45 * gamma
        core = structure_trust * l4_score + (1.0 - structure_trust) * l3_score
        core += (1.0 - meta_trust) * candidate_complexity * 0.15
    base = l1_score + l2_score + core + constraint_adjustment + 0.10 * hypothesis_alignment
    return {
        "base": base,
        "l1": l1_score,
        "l2": l2_score,
        "l3": l3_score,
        "core": core,
        "identity_trust": identity_trust,
        "constraint_adjustment": constraint_adjustment,
        "candidate_complexity": candidate_complexity,
        "meta_trust": meta_trust,
        "structure_trust": structure_trust,
        "hypothesis_alignment": hypothesis_alignment,
        "hypothesis_strength": hypothesis_strength,
        "hypothesis_name": hypothesis_name,
    }


def run_condition(tasks: list[dict], condition: str, seed: int) -> dict[str, object]:
    settings = _condition_settings(condition)
    completion = settings["completion"]
    correct = 0
    total = 0
    predictions = []
    trace_count = 0
    lineage_hits = 0
    trace_completeness = []
    decomposition_coverage_scores = []
    decomposition_ambiguity_scores = []
    decomposition_alignment_scores = []
    assembly_quality_scores = []
    promotion_bonus_scores = []
    promotion_insights = []
    first_weights = None
    last_weights = None
    rng = np.random.default_rng(seed + (10_000 if completion else 0))
    promotion_snapshot = {
        "total_occurrences": 0,
        "promoted_count": 0,
        "reused_count": 0,
        "promotion_rate": 0.0,
        "reuse_rate": 0.0,
        "retire_count": 0,
    }

    with tempfile.TemporaryDirectory(prefix=f"hpm_arc_promotion_{condition}_{seed}_") as promo_dir:
        promotion_ledger = PromotionLedger(
            Path(promo_dir) / "promotion.sqlite",
            rule=PromotionRule(
                min_occurrences=2,
                min_support=0.5,
                max_ambiguity=0.85,
                min_delta_lift=0.0,
                retention_window=6,
                selected_weight=1.0,
                unselected_weight=0.30,
            ),
        )

        for task_idx, task in enumerate(tasks):
            orch, agents, encoders = _make_structured_stack(
                seed + task_idx * 17,
                completion=completion,
                include_constraints=settings["include_constraints"],
                include_identity_snapshots=settings["include_identity_snapshots"],
                adaptive_meta_eval=settings["adaptive_meta_eval"],
            )
            l1_agents, l2_agents, l3_agents = agents
            enc1, enc2, enc3 = encoders
            level0 = orch.orches[0] if getattr(orch, "orches", None) else None
            field = getattr(level0, "field", None) if level0 is not None else None
            field_constraints = field.constraints_for(None) if field is not None and hasattr(field, "constraints_for") else []
            head = L4GenerativeHead(feature_dim_in=9, feature_dim_out=14)
            monitor = L5MetaMonitor()
            orch._completion_head = head
            orch._completion_monitor = monitor

            l1_train_vecs = []
            l2_train_vecs = []
            l3_train_vecs = []
            train_pairs = list(task["train"])
            hypotheses = _arc_hypotheses_from_train(train_pairs)
            train_decompositions = [
                extract_arc_decomposition(
                    pair["input"],
                    pair["output"],
                    task_id=f"task_{task_idx}",
                    pair_id=f"train_{pair_idx}",
                )
                for pair_idx, pair in enumerate(train_pairs)
            ]
            decomposition_profile = build_arc_decomposition_profile(train_decompositions)

            for _ in range(TRAIN_REPS):
                for pair in train_pairs:
                    step = orch.step((pair["input"], pair["output"]))
                    level3 = step.get("level3", {})
                    result = next(iter(level3.values())) if level3 else {}
                    if result:
                        trace = result.get("decision_trace", {})
                        completeness = sum(1 for part in (
                            trace.get("trace_id"),
                            trace.get("selected_pattern_ids"),
                            trace.get("selected_parent_ids"),
                            trace.get("constraint_ids"),
                            trace.get("meta_evaluator_state"),
                            trace.get("signal_source"),
                        ) if part) / 6.0
                        lineage = int(bool(trace.get("selected_parent_ids")) or bool(result.get("recombination_accepted")))
                        trace_count += 1
                        lineage_hits += lineage
                        trace_completeness.append(completeness)
                        weights = tuple(result.get("meta_evaluator_state", {}).get("weights", ()))
                        if weights:
                            if first_weights is None:
                                first_weights = weights
                            last_weights = weights

                    l2_ctx_train = (
                        _context_from_agents(l1_agents, include_identity_snapshots=settings["include_identity_snapshots"])
                        if completion
                        else (None, None, None)
                    )
                    l3_ctx_train = (
                        _context_from_agents(l2_agents, include_identity_snapshots=settings["include_identity_snapshots"])
                        if completion
                        else (None, None, None)
                    )
                    l1_vec = _mean_vecs(_encode_with_optional_context(enc1, (pair["input"], pair["output"]), None), 64)
                    l2_vec = _mean_vecs(_encode_with_optional_context(enc2, (pair["input"], pair["output"]), orch._epistemic[0], *l2_ctx_train), 9)
                    l3_vec = _mean_vecs(_encode_with_optional_context(enc3, (pair["input"], pair["output"]), orch._epistemic[1], *l3_ctx_train), 14)
                    l1_train_vecs.append(l1_vec)
                    l2_train_vecs.append(l2_vec)
                    l3_train_vecs.append(l3_vec)
                    head.accumulate(l2_vec, l3_vec)
                    monitor.update(head.predict(l2_vec), l3_vec)

            head.fit()
            orch._completion_head = head
            orch._completion_monitor = monitor

            l1_proto = _mean_vecs(l1_train_vecs, 64)
            l2_proto = _mean_vecs(l2_train_vecs, 9)
            l3_proto = _mean_vecs(l3_train_vecs, 14)

            candidates, test_input, correct_output = _candidate_bundle(task, tasks, task_idx)
            feature_rows = []
            candidate_decompositions = []
            candidate_patterns = []
            for cand_idx, cand in enumerate(candidates):
                candidate_decomp = extract_arc_decomposition(
                    test_input,
                    cand,
                    task_id=f"task_{task_idx}",
                    pair_id=f"candidate_{cand_idx}",
                )
                candidate_decompositions.append(candidate_decomp)
                best_pattern = best_promotable_pattern(candidate_decomp)
                candidate_patterns.append(best_pattern)
                promotion_signature = promotion_ledger.signature_for(best_pattern) if best_pattern is not None else ""
                promotion_bonus = 0.0
                promotion_reused = 0.0
                if completion and best_pattern is not None and promotion_signature:
                    promotion_bonus, _, _ = promotion_ledger.score_bonus(best_pattern)
                    promotion_reused = 1.0 if promotion_bonus > 0.0 else 0.0
                candidate_row = _candidate_feature_bundle(
                    orch,
                    agents,
                    encoders,
                    test_input,
                    cand,
                    completion,
                    settings["include_identity_snapshots"],
                    field_constraints,
                    hypotheses,
                    l1_proto,
                    l2_proto,
                    l3_proto,
                )
                part_alignment, part_breakdown = score_arc_decomposition(decomposition_profile, candidate_decomp)
                candidate_row.update({
                    "part_alignment": part_alignment,
                    "part_support": candidate_decomp.coverage,
                    "part_ambiguity": candidate_decomp.ambiguity_rate,
                    "part_selected_ratio": candidate_decomp.selected_count / max(candidate_decomp.composite_count, 1),
                    "part_confidence": candidate_decomp.mean_confidence,
                    "part_breakdown_coverage_gap": part_breakdown["coverage_gap"],
                    "part_breakdown_ambiguity_gap": part_breakdown["ambiguity_gap"],
                    "promotion_signature": promotion_signature,
                    "promotion_bonus": promotion_bonus,
                    "promotion_reused": promotion_reused,
                })
                feature_rows.append(candidate_row)
                promotion_bonus_scores.append(promotion_bonus)
            if completion:
                scores, score_contributions = _apply_rank_margin_reweight_with_details(feature_rows)
            else:
                scores = _apply_rank_margin_reweight(feature_rows, completion)
                score_contributions = []

            idx = int(np.argmin(scores))
            predictions.append(idx)
            predicted = candidates[idx]
            chosen_decomposition = candidate_decompositions[idx]
            chosen_row = feature_rows[idx]
            decomposition_coverage_scores.append(chosen_decomposition.coverage)
            decomposition_ambiguity_scores.append(chosen_decomposition.ambiguity_rate)
            decomposition_alignment_scores.append(float(chosen_row.get("part_alignment", 0.0)))
            assembly_quality_scores.append(float(chosen_decomposition.selected_count / max(chosen_decomposition.composite_count, 1)))
            if predicted == correct_output:
                correct += 1
            total += 1
            if completion:
                base_scores = [row.get("base", 0.0) for row in feature_rows]
                base_idx = int(np.argmin(base_scores)) if base_scores else idx
                chosen_detail = score_contributions[idx] if idx < len(score_contributions) else {}
                selected_bonus = float(chosen_row.get("promotion_bonus", 0.0))
                insight = {
                    "task_idx": task_idx,
                    "base_candidate_idx": base_idx,
                    "selected_candidate_idx": idx,
                    "promotion_signature": str(chosen_row.get("promotion_signature", "")),
                    "promotion_bonus": selected_bonus,
                    "detail": chosen_detail,
                }
                if base_idx != idx or selected_bonus > 0.0:
                    promotion_insights.append(insight)

            for cand_idx, (candidate_decomp, row, pattern) in enumerate(zip(candidate_decompositions, feature_rows, candidate_patterns)):
                if pattern is None:
                    continue
                signature = str(row.get("promotion_signature", ""))
                if not signature:
                    continue
                promotion_ledger.record_occurrence(
                    task_id=f"task_{task_idx}",
                    trace_id=candidate_decomp.trace.trace_id,
                    pattern=pattern,
                    baseline_score=float(row["base"]),
                    final_score=float(scores[cand_idx]),
                    coverage=float(candidate_decomp.coverage),
                    ambiguity_rate=float(candidate_decomp.ambiguity_rate),
                    selected=bool(cand_idx == idx),
                    source_part_ids=pattern.part_ids,
                    source_relation_ids=pattern.relation_ids,
                    promotion_bonus=float(row.get("promotion_bonus", 0.0)) if completion else 0.0,
                    persist=bool(cand_idx == idx),
                )

        promotion_snapshot = promotion_ledger.snapshot().to_dict()

    drift = 0.0
    if first_weights is not None and last_weights is not None and len(first_weights) == len(last_weights):
        drift = float(np.sum(np.abs(np.asarray(last_weights) - np.asarray(first_weights))))

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "predictions": tuple(predictions),
        "lineage_integrity": float(lineage_hits / trace_count) if trace_count else 0.0,
        "trace_completeness": float(np.mean(trace_completeness)) if trace_completeness else 0.0,
        "evaluator_drift": drift,
        "decomposition_coverage": float(np.mean(decomposition_coverage_scores)) if decomposition_coverage_scores else 0.0,
        "decomposition_ambiguity_rate": float(np.mean(decomposition_ambiguity_scores)) if decomposition_ambiguity_scores else 0.0,
        "decomposition_alignment": float(np.mean(decomposition_alignment_scores)) if decomposition_alignment_scores else 0.0,
        "assembly_quality": float(np.mean(assembly_quality_scores)) if assembly_quality_scores else 0.0,
        "promotion_total_occurrences": int(promotion_snapshot["total_occurrences"]),
        "promotion_promoted_count": int(promotion_snapshot["promoted_count"]),
        "promotion_reused_count": int(promotion_snapshot["reused_count"]),
        "promotion_rate": float(promotion_snapshot["promotion_rate"]),
        "reuse_rate": float(promotion_snapshot["reuse_rate"]),
        "promotion_retire_count": int(promotion_snapshot["retire_count"]),
        "promotion_bonus_mean": float(np.mean(promotion_bonus_scores)) if promotion_bonus_scores else 0.0,
        "promotion_insights": promotion_insights,
    }


def run_ablation_sweep(
    split: str = "evaluation",
    root: str | Path | None = None,
    max_tasks: int | None = None,
    seed: int = 42,
) -> dict[str, object]:
    all_tasks = load_tasks(root=root, split=split)
    eligible = [task for task in all_tasks if task_fits(task)]
    if max_tasks is not None:
        eligible = eligible[:max_tasks]

    results = {condition: run_condition(eligible, condition, seed) for condition in ARC_ABLATION_CONDITIONS}
    baseline = results["baseline"]
    return {
        "conditions": results,
        "delta": {
            condition: {
                "accuracy": results[condition]["accuracy"] - baseline["accuracy"],
                "lineage_integrity": results[condition]["lineage_integrity"] - baseline["lineage_integrity"],
                "trace_completeness": results[condition]["trace_completeness"] - baseline["trace_completeness"],
                "evaluator_drift": results[condition]["evaluator_drift"] - baseline["evaluator_drift"],
                "decomposition_coverage": results[condition]["decomposition_coverage"] - baseline["decomposition_coverage"],
                "decomposition_ambiguity_rate": results[condition]["decomposition_ambiguity_rate"] - baseline["decomposition_ambiguity_rate"],
                "decomposition_alignment": results[condition]["decomposition_alignment"] - baseline["decomposition_alignment"],
                "assembly_quality": results[condition]["assembly_quality"] - baseline["assembly_quality"],
                "promotion_rate": results[condition]["promotion_rate"] - baseline["promotion_rate"],
                "reuse_rate": results[condition]["reuse_rate"] - baseline["reuse_rate"],
                "promotion_promoted_count": results[condition]["promotion_promoted_count"] - baseline["promotion_promoted_count"],
                "promotion_bonus_mean": results[condition]["promotion_bonus_mean"] - baseline["promotion_bonus_mean"],
            }
            for condition in ARC_ABLATION_CONDITIONS
            if condition != "baseline"
        },
        "tasks_run": len(eligible),
        "excluded": len(all_tasks) - len(eligible),
        "split": split,
        "stack_levels": 5,
    }


def run(split: str = "evaluation", root: str | Path | None = None, max_tasks: int | None = None, seed: int = 42) -> dict[str, dict[str, object]]:
    all_tasks = load_tasks(root=root, split=split)
    eligible = [task for task in all_tasks if task_fits(task)]
    if max_tasks is not None:
        eligible = eligible[:max_tasks]

    baseline = run_condition(eligible, "baseline", seed)
    completion = run_condition(eligible, "completion", seed)
    return {
        "baseline": baseline,
        "completion": completion,
        "delta": {
            "accuracy": completion["accuracy"] - baseline["accuracy"],
            "lineage_integrity": completion["lineage_integrity"] - baseline["lineage_integrity"],
            "trace_completeness": completion["trace_completeness"] - baseline["trace_completeness"],
            "evaluator_drift": completion["evaluator_drift"] - baseline["evaluator_drift"],
            "decomposition_coverage": completion["decomposition_coverage"] - baseline["decomposition_coverage"],
            "decomposition_ambiguity_rate": completion["decomposition_ambiguity_rate"] - baseline["decomposition_ambiguity_rate"],
            "decomposition_alignment": completion["decomposition_alignment"] - baseline["decomposition_alignment"],
            "assembly_quality": completion["assembly_quality"] - baseline["assembly_quality"],
            "promotion_rate": completion["promotion_rate"] - baseline["promotion_rate"],
            "reuse_rate": completion["reuse_rate"] - baseline["reuse_rate"],
            "promotion_promoted_count": completion["promotion_promoted_count"] - baseline["promotion_promoted_count"],
            "promotion_bonus_mean": completion["promotion_bonus_mean"] - baseline["promotion_bonus_mean"],
        },
        "tasks_run": len(eligible),
        "excluded": len(all_tasks) - len(eligible),
        "split": split,
        "stack_levels": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC-AGI-2 comparison benchmark for baseline vs completion-aware structured agents")
    parser.add_argument("--split", choices=("training", "evaluation"), default="evaluation")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation-sweep", action="store_true", help="Run baseline plus completion ablations")
    args = parser.parse_args()

    if args.ablation_sweep:
        print("Running ARC-AGI-2 ablation sweep...", flush=True)
        result = run_ablation_sweep(split=args.split, root=args.root, max_tasks=args.max_tasks, seed=args.seed)
        rows = []
        for condition in ARC_ABLATION_CONDITIONS:
            cur = result["conditions"][condition]
            rows.append(
                {
                    "Condition": condition,
                    "Accuracy": f"{cur['accuracy']:.1%}",
                    "Lineage": f"{cur['lineage_integrity']:.3f}",
                    "Trace": f"{cur['trace_completeness']:.3f}",
                    "Drift": f"{cur['evaluator_drift']:.3f}",
                    "PartCov": f"{cur['decomposition_coverage']:.3f}",
                    "Ambig": f"{cur['decomposition_ambiguity_rate']:.3f}",
                    "Align": f"{cur['decomposition_alignment']:.3f}",
                    "Assem": f"{cur['assembly_quality']:.3f}",
                    "Promo": f"{cur['promotion_rate']:.3f}",
                    "Reuse": f"{cur['reuse_rate']:.3f}",
                }
            )
        print_results_table(
            title=f"ARC-AGI-2 Ablation Sweep ({result['split']})",
            cols=["Condition", "Accuracy", "Lineage", "Trace", "Drift", "PartCov", "Ambig", "Align", "Assem", "Promo", "Reuse"],
            rows=rows,
        )
        print(f"Tasks: {result['tasks_run']} evaluated, {result['excluded']} excluded")
        print(f"Stack levels: {result['stack_levels']}")
        for condition, delta in result["delta"].items():
            print(
                f"Delta vs baseline [{condition}]: "
                f"accuracy={delta['accuracy']:+.3f} "
                f"lineage={delta['lineage_integrity']:+.3f} "
                f"trace={delta['trace_completeness']:+.3f} "
                f"drift={delta['evaluator_drift']:+.3f} "
                f"partcov={delta['decomposition_coverage']:+.3f} "
                f"ambig={delta['decomposition_ambiguity_rate']:+.3f} "
                f"align={delta['decomposition_alignment']:+.3f} "
                f"assem={delta['assembly_quality']:+.3f} "
                f"promo={delta['promotion_rate']:+.3f} "
                f"reuse={delta['reuse_rate']:+.3f}"
            )
        return

    print("Running ARC-AGI-2 completion comparison...", flush=True)
    result = run(split=args.split, root=args.root, max_tasks=args.max_tasks, seed=args.seed)
    base = result["baseline"]
    comp = result["completion"]

    print_results_table(
        title=f"ARC-AGI-2 Comparison ({result['split']})",
        cols=["Condition", "Accuracy", "Lineage", "Trace", "Drift", "PartCov", "Ambig", "Align", "Assem", "Promo", "Reuse"],
        rows=[
            {"Condition": "baseline", "Accuracy": f"{base['accuracy']:.1%}", "Lineage": f"{base['lineage_integrity']:.3f}", "Trace": f"{base['trace_completeness']:.3f}", "Drift": f"{base['evaluator_drift']:.3f}", "PartCov": f"{base['decomposition_coverage']:.3f}", "Ambig": f"{base['decomposition_ambiguity_rate']:.3f}", "Align": f"{base['decomposition_alignment']:.3f}", "Assem": f"{base['assembly_quality']:.3f}"},
            {"Condition": "completion", "Accuracy": f"{comp['accuracy']:.1%}", "Lineage": f"{comp['lineage_integrity']:.3f}", "Trace": f"{comp['trace_completeness']:.3f}", "Drift": f"{comp['evaluator_drift']:.3f}", "PartCov": f"{comp['decomposition_coverage']:.3f}", "Ambig": f"{comp['decomposition_ambiguity_rate']:.3f}", "Align": f"{comp['decomposition_alignment']:.3f}", "Assem": f"{comp['assembly_quality']:.3f}"},
        ],
    )
    print(f"Tasks: {result['tasks_run']} evaluated, {result['excluded']} excluded")
    print(f"Stack levels: {result['stack_levels']}")
    print(f"Delta: accuracy={result['delta']['accuracy']:+.3f} lineage={result['delta']['lineage_integrity']:+.3f} trace={result['delta']['trace_completeness']:+.3f} drift={result['delta']['evaluator_drift']:+.3f} partcov={result['delta']['decomposition_coverage']:+.3f} ambig={result['delta']['decomposition_ambiguity_rate']:+.3f} align={result['delta']['decomposition_alignment']:+.3f} assem={result['delta']['assembly_quality']:+.3f} promo={result['delta']['promotion_rate']:+.3f} reuse={result['delta']['reuse_rate']:+.3f}")


if __name__ == "__main__":
    main()
