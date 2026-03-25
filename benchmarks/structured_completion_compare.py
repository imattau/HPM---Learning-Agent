"""Structured completion comparison benchmark."""

from __future__ import annotations

import argparse
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.multi_agent_arc import ensemble_score
from benchmarks.multi_agent_common import make_orchestrator
from benchmarks.structured_math import generate_tasks
from hpm.agents.hierarchical import extract_relational_bundle
from hpm.agents.structured import StructuredOrchestrator
from hpm.completion_types import FieldConstraint
from hpm.encoders.math_encoders import MathL1Encoder, MathL2Encoder, MathL3Encoder

TRAIN_REPS = 5


@dataclass(frozen=True)
class Settings:
    train_reps: int = TRAIN_REPS
    score_noise_std: float = 0.0
    completion_noise_reduction: float = 0.0
    extra_candidates: int = 0


class CompletionL2(MathL2Encoder):
    def encode(self, observation: tuple, epistemic=None, relational_bundles=None, structural_messages=None, identity_snapshots=None):
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
            v[8] = float(np.clip(v[8] + 0.15 * msg, 0.0, 1.0))
            v[9] = float(np.clip(v[9] + 0.10 * lineage, 0.0, 1.0))
            out.append(v)
        return out


class CompletionL3(MathL3Encoder):
    def encode(self, observation: tuple, epistemic=None, relational_bundles=None, structural_messages=None, identity_snapshots=None):
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
            v[10] = float(np.clip(v[10] + 0.20 * rel + 0.10 * lineage, 0.0, 1.0))
            v[11] = float(np.clip(v[11] + 0.20 * msg + 0.10 * lineage, 0.0, 1.0))
            out.append(v)
        return out


def _encode_with_optional_context(encoder, observation, epistemic, relational_bundles, structural_messages, identity_snapshots):
    params = inspect.signature(encoder.encode).parameters
    kwargs = {"epistemic": epistemic}
    if "relational_bundles" in params:
        kwargs["relational_bundles"] = relational_bundles
    if "structural_messages" in params:
        kwargs["structural_messages"] = structural_messages
    if "identity_snapshots" in params:
        kwargs["identity_snapshots"] = identity_snapshots
    return encoder.encode(observation, **kwargs)


def _make_orch(condition: str, seed: int):
    completion = condition == "completion"
    agent_kwargs = dict(seed=seed, T_recomb=1, recomb_cooldown=1, min_recomb_level=1, kappa_max=1.0, N_recomb=1, conflict_threshold=2.0)

    l1_orch, l1_agents, _ = make_orchestrator(2, 14, ["l1_0", "l1_1"], with_monitor=False, gamma_soc=0.5, init_sigma=2.0, agent_seeds=[seed + 1, seed + 2], **agent_kwargs)
    l2_orch, l2_agents, _ = make_orchestrator(2, 10, ["l2_0", "l2_1"], with_monitor=False, gamma_soc=0.5, init_sigma=2.0, agent_seeds=[seed + 11, seed + 12], **agent_kwargs)
    l3_orch, l3_agents, _ = make_orchestrator(1, 12, ["l3_0"], with_monitor=False, gamma_soc=0.5, init_sigma=2.0, agent_seeds=[seed + 21], **agent_kwargs)

    if completion:
        for orch in (l1_orch, l2_orch, l3_orch):
            if orch.field is not None:
                orch.field.add_constraint(FieldConstraint("prefer_simple", "*", 0.6, "benchmark", seed))
                orch.field.add_constraint(FieldConstraint("penalize_complexity", "*", 0.4, "benchmark", seed))

    enc2 = CompletionL2() if completion else MathL2Encoder()
    enc3 = CompletionL3() if completion else MathL3Encoder()
    return StructuredOrchestrator(
        encoders=[MathL1Encoder(), enc2, enc3],
        orches=[l1_orch, l2_orch, l3_orch],
        agents=[l1_agents, l2_agents, l3_agents],
        level_Ks=[1, 1, 2],
        relational_bundles_enabled=completion,
        structural_messages_to_encoders_enabled=completion,
        identity_snapshots_to_encoders_enabled=completion,
    ), l2_agents, l3_agents, enc3


def _level3_result(step_result: dict) -> dict:
    level3 = step_result.get("level3", {})
    return next(iter(level3.values())) if level3 else {}


def _trace_score(result: dict) -> tuple[float, int]:
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
    return completeness, lineage


def _score_candidate(orch, l2_agents, l3_agents, l3_enc, test_input, candidate, completion: bool, rng: np.random.Generator, settings: Settings) -> float:
    epistemic = orch._epistemic[1] if len(orch._epistemic) > 1 else None
    rel = [extract_relational_bundle(agent) for agent in l2_agents] if completion else None
    msgs = [] if completion else None
    if completion:
        for agent in l2_agents:
            msgs.extend(agent.consume_structural_inbox(clear=False))
    snapshots = [] if completion else None
    if completion:
        for agent in l2_agents:
            lifecycle = getattr(agent, "_lifecycle", None)
            snapshots.append(lifecycle.snapshot() if lifecycle is not None and hasattr(lifecycle, "snapshot") else {})
    vecs = _encode_with_optional_context(l3_enc, (test_input, candidate), epistemic, rel, msgs, snapshots)
    if not vecs:
        return float("inf")
    vec = np.mean(vecs, axis=0)
    if settings.score_noise_std > 0.0:
        attenuation = 1.0 - (settings.completion_noise_reduction if completion else 0.0)
        vec = vec + rng.normal(0.0, settings.score_noise_std * float(np.clip(attenuation, 0.2, 1.0)), size=vec.shape)
    return ensemble_score(l3_agents, vec)


def _augment_candidates(candidates, test_output, extra: int):
    if extra <= 0:
        return list(candidates)
    out = list(candidates)
    try:
        base = float(test_output)
    except Exception:
        return out
    for delta in [1, -1, 2, -2, 3, -3, 0.5, -0.5]:
        if len(out) >= len(candidates) + extra:
            break
        probe = sympy.sympify(base + delta)
        if all(sympy.simplify(probe - existing) != 0 for existing in out):
            out.append(probe)
    return out


def run_condition(tasks: list[dict], condition: str, settings: Settings, seed: int) -> dict[str, object]:
    completion = condition == "completion"
    correct = 0
    total = 0
    predictions = []
    trace_count = 0
    lineage_hits = 0
    trace_completeness = []
    first_weights = None
    last_weights = None
    rng = np.random.default_rng(seed + (10_000 if completion else 0))

    for task in tasks:
        orch, l2_agents, l3_agents, l3_enc = _make_orch(condition, seed)
        for _ in range(settings.train_reps):
            for pair in task["train"]:
                step = orch.step(pair)
                result = _level3_result(step)
                if result:
                    completeness, lineage = _trace_score(result)
                    trace_count += 1
                    lineage_hits += lineage
                    trace_completeness.append(completeness)
                    weights = tuple(result.get("meta_evaluator_state", {}).get("weights", ()))
                    if weights:
                        if first_weights is None:
                            first_weights = weights
                        last_weights = weights

        candidates = _augment_candidates(task["candidates"], task["test_output"], settings.extra_candidates)
        scores = [
            _score_candidate(orch, l2_agents, l3_agents, l3_enc, task["test_input"], cand, completion, rng, settings)
            for cand in candidates
        ]
        idx = int(np.argmin(scores))
        predictions.append(idx)
        predicted = candidates[idx]
        try:
            if sympy.simplify(predicted - task["test_output"]) == 0:
                correct += 1
        except Exception:
            if predicted == task["test_output"]:
                correct += 1
        total += 1

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
    }


def run_repeatability(tasks: list[dict], condition: str, settings: Settings, seed: int) -> tuple[dict[str, object], float]:
    first = run_condition(tasks, condition, settings, seed)
    second = run_condition(tasks, condition, settings, seed)
    stability = 1.0 if first["predictions"] == second["predictions"] else 0.0
    first = dict(first)
    first["seed_stability"] = stability
    return first, stability


def main():
    parser = argparse.ArgumentParser(description="Structured Completion Comparison")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n_tasks", type=int, default=0)
    parser.add_argument("--n_per_family", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", choices=["normal", "hard"], default="normal")
    parser.add_argument("--hard_train_reps", type=int, default=2)
    parser.add_argument("--hard_extra_candidates", type=int, default=4)
    parser.add_argument("--hard_noise_std", type=float, default=0.08)
    parser.add_argument("--hard_completion_reduction", type=float, default=0.35)
    args = parser.parse_args()

    n_per_family = 2 if args.smoke else args.n_tasks or args.n_per_family
    tasks = generate_tasks(n_per_family=n_per_family, seed=args.seed)
    settings = Settings(
        train_reps=args.hard_train_reps if args.difficulty == "hard" else TRAIN_REPS,
        score_noise_std=args.hard_noise_std if args.difficulty == "hard" else 0.0,
        completion_noise_reduction=args.hard_completion_reduction if args.difficulty == "hard" else 0.0,
        extra_candidates=args.hard_extra_candidates if args.difficulty == "hard" else 0,
    )

    print(f"\nStructured Completion Comparison — {len(tasks)} tasks, seed={args.seed}, difficulty={args.difficulty}, train_reps={settings.train_reps}")
    if args.difficulty == "hard":
        print(f"Hard-mode settings: extra_candidates={settings.extra_candidates}, noise_std={settings.score_noise_std:.3f}, completion_reduction={settings.completion_noise_reduction:.2f}")
    print(f"{'Condition':<14} {'Accuracy':>10} {'SeedStable':>11} {'Lineage':>10} {'Drift':>10} {'TraceCmp':>10} {'Correct':>10} {'Total':>8}")
    print("-" * 84)

    baseline, base_stability = run_repeatability(tasks, "baseline", settings, args.seed)
    completion, comp_stability = run_repeatability(tasks, "completion", settings, args.seed)
    baseline["seed_stability"] = base_stability
    completion["seed_stability"] = comp_stability

    for name, m in (("baseline", baseline), ("completion", completion)):
        print(f"{name:<14} {m['accuracy']:>10.3f} {m['seed_stability']:>11.3f} {m['lineage_integrity']:>10.3f} {m['evaluator_drift']:>10.3f} {m['trace_completeness']:>10.3f} {m['correct']:>10} {m['total']:>8}")

    print("\nDeltas vs baseline:")
    print(f"  accuracy          {completion['accuracy'] - baseline['accuracy']:+.3f}")
    print(f"  seed_stability    {comp_stability - base_stability:+.3f}")
    print(f"  lineage_integrity {completion['lineage_integrity'] - baseline['lineage_integrity']:+.3f}")
    print(f"  evaluator_drift   {completion['evaluator_drift'] - baseline['evaluator_drift']:+.3f}")
    print(f"  trace_completeness {completion['trace_completeness'] - baseline['trace_completeness']:+.3f}")


if __name__ == "__main__":
    main()
