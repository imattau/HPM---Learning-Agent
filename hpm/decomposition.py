
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TaskAtom:
    atom_id: str
    kind: str
    features: dict[str, Any]
    confidence: float
    source: str
    trace_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskPart:
    part_id: str
    label: str
    atom_ids: tuple[str, ...]
    features: dict[str, Any]
    confidence: float
    parent_part_ids: tuple[str, ...] = ()
    trace_id: str = ""
    ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PartRelation:
    source_part_id: str
    target_part_id: str
    relation_type: str
    weight: float
    evidence: dict[str, Any]
    trace_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompositePattern:
    pattern_id: str
    part_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    assembly_rule: str
    score: float
    stability: float
    parent_pattern_ids: tuple[str, ...]
    trace_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HypothesisCandidate:
    hypothesis_id: str
    hypothesis_type: str
    source_part_ids: tuple[str, ...]
    predicted_transform: dict[str, Any]
    support: float
    novelty: float
    confidence: float
    trace_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssemblyTrace:
    trace_id: str
    task_id: str
    extracted_parts: tuple[str, ...]
    selected_relations: tuple[str, ...]
    candidate_patterns: tuple[str, ...]
    selected_pattern_ids: tuple[str, ...]
    rejection_reason: str
    score_breakdown: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskDecompositionResult:
    trace: AssemblyTrace
    atoms: tuple[TaskAtom, ...]
    parts: tuple[TaskPart, ...]
    relations: tuple[PartRelation, ...]
    candidate_patterns: tuple[CompositePattern, ...]
    hypotheses: tuple[HypothesisCandidate, ...]
    coverage: float
    ambiguity_rate: float
    part_count: int
    relation_count: int
    composite_count: int
    selected_count: int
    rejected_count: int
    mean_confidence: float
    assigned_atom_ids: tuple[str, ...]
    unassigned_atom_ids: tuple[str, ...]
    ambiguous_part_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace": self.trace.to_dict(),
            "atoms": [atom.to_dict() for atom in self.atoms],
            "parts": [part.to_dict() for part in self.parts],
            "relations": [relation.to_dict() for relation in self.relations],
            "candidate_patterns": [pattern.to_dict() for pattern in self.candidate_patterns],
            "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
            "coverage": float(self.coverage),
            "ambiguity_rate": float(self.ambiguity_rate),
            "part_count": int(self.part_count),
            "relation_count": int(self.relation_count),
            "composite_count": int(self.composite_count),
            "selected_count": int(self.selected_count),
            "rejected_count": int(self.rejected_count),
            "mean_confidence": float(self.mean_confidence),
            "assigned_atom_ids": list(self.assigned_atom_ids),
            "unassigned_atom_ids": list(self.unassigned_atom_ids),
            "ambiguous_part_ids": list(self.ambiguous_part_ids),
        }


@dataclass(frozen=True)
class ArcDecompositionProfile:
    coverage: float
    ambiguity_rate: float
    part_count: float
    relation_count: float
    composite_count: float
    selected_count: float
    mean_confidence: float
    rejection_count: float
    mean_alignment: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _grid_signature(grid: list[list[int]]) -> dict[str, float | tuple[int, int]]:
    arr = np.asarray(grid, dtype=int)
    if arr.size == 0:
        return {
            "shape": (0, 0),
            "colors": 0.0,
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


def _pair_effect_label(input_obj, output_obj, move_distance: float, area_ratio: float) -> str:
    color_changed = input_obj.color != output_obj.color
    if not color_changed and move_distance < 0.5 and abs(area_ratio - 1.0) < 0.15:
        return "identity"
    if not color_changed and move_distance >= 0.5 and abs(area_ratio - 1.0) < 0.25:
        return "translation"
    if color_changed and move_distance < 0.5 and abs(area_ratio - 1.0) < 0.25:
        return "color_change"
    if abs(area_ratio - 1.0) >= 0.25 or move_distance >= 1.25:
        return "structure_transform"
    return "mixed_transform"


def _pair_confidence(move_distance: float, area_ratio: float, color_changed: bool) -> float:
    move_penalty = min(move_distance / 6.0, 1.0)
    area_penalty = min(abs(area_ratio - 1.0), 1.0)
    color_penalty = 0.18 if color_changed else 0.0
    confidence = 1.0 - (0.45 * move_penalty + 0.35 * area_penalty + color_penalty)
    return float(np.clip(confidence, 0.05, 1.0))


def _atom_features(obj, source: str) -> dict[str, Any]:
    min_r, min_c, max_r, max_c = obj.bbox
    return {
        "source": source,
        "color": int(obj.color),
        "area": int(obj.area),
        "perimeter": int(obj.perimeter),
        "bbox": (int(min_r), int(min_c), int(max_r), int(max_c)),
        "centroid": (float(obj.centroid[0]), float(obj.centroid[1])),
    }


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def extract_arc_decomposition(
    input_grid: list[list[int]],
    output_grid: list[list[int]],
    *,
    task_id: str,
    pair_id: str,
    trace_id: str | None = None,
) -> TaskDecompositionResult:
    """Decompose an ARC pair into atoms, parts, relations, and composites."""
    if trace_id is None:
        trace_id = f"{task_id}:{pair_id}"

    from benchmarks.arc_encoders import match_objects, parse_objects

    input_objects = parse_objects(input_grid)
    output_objects = parse_objects(output_grid)
    matched_pairs, appeared, disappeared = match_objects(input_objects, output_objects)

    input_color_counts: dict[int, int] = {}
    output_color_counts: dict[int, int] = {}
    for obj in input_objects:
        input_color_counts[obj.color] = input_color_counts.get(obj.color, 0) + 1
    for obj in output_objects:
        output_color_counts[obj.color] = output_color_counts.get(obj.color, 0) + 1

    atoms: list[TaskAtom] = []
    parts: list[TaskPart] = []
    relations: list[PartRelation] = []
    candidate_patterns: list[CompositePattern] = []
    hypotheses: list[HypothesisCandidate] = []

    for obj in input_objects:
        atom_id = f"{trace_id}:atom:in:{obj.id}"
        atoms.append(TaskAtom(atom_id, "input_object", _atom_features(obj, "input"), 1.0, "input", trace_id))
    for obj in output_objects:
        atom_id = f"{trace_id}:atom:out:{obj.id}"
        atoms.append(TaskAtom(atom_id, "output_object", _atom_features(obj, "output"), 1.0, "output", trace_id))

    root_part_id = f"{trace_id}:part:task_root"
    root_part = TaskPart(
        part_id=root_part_id,
        label="task_root",
        atom_ids=tuple(),
        features={"kind": "summary"},
        confidence=1.0,
        parent_part_ids=(),
        trace_id=trace_id,
        ambiguous=False,
    )
    parts.append(root_part)

    relation_ids: list[str] = []
    effect_groups: dict[str, list[str]] = {}
    effect_confidences: dict[str, list[float]] = {}
    effect_support: dict[str, list[float]] = {}
    assigned_atom_ids: set[str] = set()
    ambiguous_part_ids: set[str] = set()

    def add_relation(source: str, target: str, relation_type: str, weight: float, evidence: dict[str, Any]) -> str:
        relation_id = f"{trace_id}:rel:{len(relations)}"
        relations.append(PartRelation(source, target, relation_type, float(weight), evidence, trace_id))
        relation_ids.append(relation_id)
        return relation_id

    for pair_index, pair in enumerate(matched_pairs):
        move_distance = float(np.hypot(
            pair.output_obj.centroid[0] - pair.input_obj.centroid[0],
            pair.output_obj.centroid[1] - pair.input_obj.centroid[1],
        ))
        input_area = max(float(pair.input_obj.area), 1.0)
        area_ratio = float(pair.output_obj.area) / input_area
        color_changed = pair.input_obj.color != pair.output_obj.color
        effect_label = _pair_effect_label(pair.input_obj, pair.output_obj, move_distance, area_ratio)
        confidence = _pair_confidence(move_distance, area_ratio, color_changed)
        ambiguous = (
            input_color_counts.get(pair.input_obj.color, 0) > 1
            or output_color_counts.get(pair.output_obj.color, 0) > 1
            or effect_label == "mixed_transform"
            or confidence < 0.55
        )
        part_id = f"{trace_id}:part:match:{pair_index}"
        atom_ids = (
            f"{trace_id}:atom:in:{pair.input_obj.id}",
            f"{trace_id}:atom:out:{pair.output_obj.id}",
        )
        part = TaskPart(
            part_id=part_id,
            label=effect_label,
            atom_ids=atom_ids,
            features={
                "effect_label": effect_label,
                "move_distance": move_distance,
                "area_ratio": area_ratio,
                "color_changed": float(color_changed),
                "input_color": float(pair.input_obj.color),
                "output_color": float(pair.output_obj.color),
            },
            confidence=confidence,
            parent_part_ids=(root_part_id,),
            trace_id=trace_id,
            ambiguous=ambiguous,
        )
        parts.append(part)
        assigned_atom_ids.update(atom_ids)
        if ambiguous:
            ambiguous_part_ids.add(part_id)
        effect_groups.setdefault(effect_label, []).append(part_id)
        effect_confidences.setdefault(effect_label, []).append(confidence)
        effect_support.setdefault(effect_label, []).append(1.0)

        add_relation(root_part_id, part_id, "contains", confidence, {"effect": effect_label})
        if effect_label == "identity":
            add_relation(root_part_id, part_id, "preserves", confidence, {"move_distance": move_distance})
        elif effect_label == "translation":
            add_relation(root_part_id, part_id, "moves", confidence, {"move_distance": move_distance})
        elif effect_label == "color_change":
            add_relation(root_part_id, part_id, "changes_color", confidence, {"area_ratio": area_ratio})
        elif effect_label == "structure_transform":
            add_relation(root_part_id, part_id, "transforms", confidence, {"move_distance": move_distance, "area_ratio": area_ratio})
        else:
            add_relation(root_part_id, part_id, "ambiguous_transform", confidence, {"move_distance": move_distance, "area_ratio": area_ratio})

    for pair_a in parts[1:]:
        for pair_b in parts[1:]:
            if pair_a.part_id >= pair_b.part_id:
                continue
            effect_a = pair_a.label
            effect_b = pair_b.label
            if effect_a == effect_b and effect_a != "mixed_transform":
                score = 0.5 * (pair_a.confidence + pair_b.confidence)
                add_relation(pair_a.part_id, pair_b.part_id, "same_transform", score, {"effect": effect_a})

    total_atoms = len(atoms)
    assigned_count = len(assigned_atom_ids)
    coverage = float(assigned_count / total_atoms) if total_atoms else 1.0
    ambiguity_rate = float(len(ambiguous_part_ids) / max(len(parts) - 1, 1))

    for effect_label, part_ids in effect_groups.items():
        confidences = effect_confidences.get(effect_label, [])
        support = len(part_ids) / max(len(parts) - 1, 1)
        mean_confidence = _mean_or_zero(confidences)
        score = float(np.clip(0.55 * mean_confidence + 0.45 * support, 0.0, 1.0))
        candidate_id = f"{trace_id}:pattern:{effect_label}"
        candidate_patterns.append(
            CompositePattern(
                pattern_id=candidate_id,
                part_ids=tuple(part_ids),
                relation_ids=tuple(
                    relation_id
                    for relation_id, relation in zip(relation_ids, relations)
                    if relation.source_part_id in part_ids or relation.target_part_id in part_ids
                ),
                assembly_rule=f"effect::{effect_label}",
                score=score,
                stability=float(np.clip(coverage * score, 0.0, 1.0)),
                parent_pattern_ids=(root_part_id,),
                trace_id=trace_id,
            )
        )
        hypotheses.append(
            HypothesisCandidate(
                hypothesis_id=candidate_id,
                hypothesis_type=effect_label,
                source_part_ids=tuple(part_ids),
                predicted_transform={
                    "effect_label": effect_label,
                    "part_count": len(part_ids),
                    "mean_confidence": mean_confidence,
                    "support": support,
                },
                support=support,
                novelty=float(np.clip(1.0 - support, 0.0, 1.0)),
                confidence=mean_confidence,
                trace_id=trace_id,
            )
        )

    candidate_patterns.sort(key=lambda item: (-item.score, item.pattern_id))
    selected_patterns = tuple(pattern.pattern_id for pattern in candidate_patterns if pattern.score >= 0.55)
    rejected_count = sum(1 for pattern in candidate_patterns if pattern.pattern_id not in selected_patterns)
    if not selected_patterns and candidate_patterns:
        selected_patterns = (candidate_patterns[0].pattern_id,)
        rejected_count = max(0, len(candidate_patterns) - 1)

    task_composite = CompositePattern(
        pattern_id=f"{trace_id}:pattern:task_composite",
        part_ids=tuple(part.part_id for part in parts if part.part_id != root_part_id),
        relation_ids=tuple(relation_ids),
        assembly_rule="task_summary",
        score=float(np.clip(coverage, 0.0, 1.0)),
        stability=float(np.clip(1.0 - ambiguity_rate, 0.0, 1.0)),
        parent_pattern_ids=tuple(pattern.pattern_id for pattern in candidate_patterns[:2]),
        trace_id=trace_id,
    )
    candidate_patterns.insert(0, task_composite)
    selected_patterns = tuple(dict.fromkeys((task_composite.pattern_id, *selected_patterns)))
    rejected_count = max(0, len(candidate_patterns) - len(selected_patterns))

    mean_confidence = _mean_or_zero([part.confidence for part in parts if part.part_id != root_part_id])
    trace = AssemblyTrace(
        trace_id=trace_id,
        task_id=task_id,
        extracted_parts=tuple(part.part_id for part in parts if part.part_id != root_part_id),
        selected_relations=tuple(relation_ids),
        candidate_patterns=tuple(pattern.pattern_id for pattern in candidate_patterns),
        selected_pattern_ids=selected_patterns,
        rejection_reason="low_support" if len(selected_patterns) <= 1 else "mixed_support",
        score_breakdown={
            "coverage": coverage,
            "ambiguity_rate": ambiguity_rate,
            "part_count": float(len(parts) - 1),
            "relation_count": float(len(relations)),
            "composite_count": float(len(candidate_patterns)),
            "mean_confidence": mean_confidence,
        },
    )

    unassigned_atom_ids = tuple(
        atom.atom_id for atom in atoms if atom.atom_id not in assigned_atom_ids
    )
    return TaskDecompositionResult(
        trace=trace,
        atoms=tuple(atoms),
        parts=tuple(parts),
        relations=tuple(relations),
        candidate_patterns=tuple(candidate_patterns),
        hypotheses=tuple(sorted(hypotheses, key=lambda item: (-item.confidence, item.hypothesis_type))),
        coverage=coverage,
        ambiguity_rate=ambiguity_rate,
        part_count=len(parts) - 1,
        relation_count=len(relations),
        composite_count=len(candidate_patterns),
        selected_count=len(selected_patterns),
        rejected_count=rejected_count,
        mean_confidence=mean_confidence,
        assigned_atom_ids=tuple(sorted(assigned_atom_ids)),
        unassigned_atom_ids=tuple(sorted(unassigned_atom_ids)),
        ambiguous_part_ids=tuple(sorted(ambiguous_part_ids)),
    )


def build_arc_decomposition_profile(results: list[TaskDecompositionResult]) -> ArcDecompositionProfile:
    if not results:
        return ArcDecompositionProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    alignments = [score_arc_decomposition(ArcDecompositionProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), result)[0] for result in results]
    total = float(len(results))
    coverage = float(np.mean([result.coverage for result in results]))
    ambiguity = float(np.mean([result.ambiguity_rate for result in results]))
    part_count = float(np.mean([result.part_count for result in results]))
    relation_count = float(np.mean([result.relation_count for result in results]))
    composite_count = float(np.mean([result.composite_count for result in results]))
    selected_count = float(np.mean([result.selected_count for result in results]))
    mean_confidence = float(np.mean([result.mean_confidence for result in results]))
    rejection_count = float(np.mean([result.rejected_count for result in results]))
    mean_alignment = float(np.mean(alignments)) if alignments else 0.0
    return ArcDecompositionProfile(
        coverage=coverage,
        ambiguity_rate=ambiguity,
        part_count=part_count,
        relation_count=relation_count,
        composite_count=composite_count,
        selected_count=selected_count,
        mean_confidence=mean_confidence,
        rejection_count=rejection_count,
        mean_alignment=mean_alignment,
    )


def score_arc_decomposition(profile: ArcDecompositionProfile, result: TaskDecompositionResult) -> tuple[float, dict[str, float]]:
    coverage_gap = abs(result.coverage - profile.coverage)
    ambiguity_gap = abs(result.ambiguity_rate - profile.ambiguity_rate)
    part_gap = abs(result.part_count - profile.part_count) / max(profile.part_count, 1.0)
    relation_gap = abs(result.relation_count - profile.relation_count) / max(profile.relation_count, 1.0)
    composite_gap = abs(result.composite_count - profile.composite_count) / max(profile.composite_count, 1.0)
    selected_gap = abs(result.selected_count - profile.selected_count) / max(profile.selected_count, 1.0)
    confidence_gap = abs(result.mean_confidence - profile.mean_confidence)
    rejection_gap = abs(result.rejected_count - profile.rejection_count) / max(profile.rejection_count, 1.0)
    score = float(
        0.22 * coverage_gap
        + 0.16 * ambiguity_gap
        + 0.16 * part_gap
        + 0.10 * relation_gap
        + 0.10 * composite_gap
        + 0.12 * selected_gap
        + 0.14 * confidence_gap
        + 0.10 * rejection_gap
    )
    breakdown = {
        "coverage_gap": coverage_gap,
        "ambiguity_gap": ambiguity_gap,
        "part_gap": part_gap,
        "relation_gap": relation_gap,
        "composite_gap": composite_gap,
        "selected_gap": selected_gap,
        "confidence_gap": confidence_gap,
        "rejection_gap": rejection_gap,
    }
    return score, breakdown


__all__ = [
    "TaskAtom",
    "TaskPart",
    "PartRelation",
    "CompositePattern",
    "HypothesisCandidate",
    "AssemblyTrace",
    "TaskDecompositionResult",
    "ArcDecompositionProfile",
    "extract_arc_decomposition",
    "build_arc_decomposition_profile",
    "score_arc_decomposition",
]
