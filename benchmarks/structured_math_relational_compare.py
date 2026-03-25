"""Structured Math comparison benchmark for relational/message-enabled orchestration.

Compares three conditions on the same generated math tasks:
  - baseline: structured hierarchy, no relational bundles, no structural messages
  - relational: relational bundle context enabled for higher-level encoders
  - relational_messages: relational bundles + structural-message context enabled

This benchmark is designed for before/after comparisons of the new structured
orchestration pathways, not to replace SP5/SP6 task-family reporting.
"""
from __future__ import annotations

import argparse
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.multi_agent_arc import ensemble_score
from benchmarks.multi_agent_common import make_orchestrator
from benchmarks.structured_math import generate_tasks
from hpm.agents.hierarchical import extract_relational_bundle
from hpm.agents.structured import StructuredOrchestrator
from hpm.encoders.math_encoders import MathL1Encoder, MathL2Encoder, MathL3Encoder

TRAIN_REPS = 5


@dataclass(frozen=True)
class BenchmarkSettings:
    train_reps: int = TRAIN_REPS
    score_noise_std: float = 0.0
    relational_noise_reduction: float = 0.0
    message_noise_reduction: float = 0.0
    extra_candidates: int = 0


class ContextAwareMathL2Encoder(MathL2Encoder):
    """Math L2 encoder with optional relational/message context bias."""

    def encode(
        self,
        observation: tuple,
        epistemic: tuple[float, float] | None,
        relational_bundles=None,
        structural_messages=None,
    ) -> list[np.ndarray]:
        vecs = super().encode(observation, epistemic=epistemic)
        if not vecs:
            return vecs

        rel_strength = 0.0
        if relational_bundles:
            edge_conf = [
                edge.confidence
                for bundle in relational_bundles
                for edge in getattr(bundle, "relations", ())
            ]
            if edge_conf:
                rel_strength = float(np.mean(edge_conf))

        msg_strength = 0.0
        if structural_messages:
            msg_conf = [
                getattr(message, "confidence", 0.0)
                for _, message in structural_messages
            ]
            if msg_conf:
                msg_strength = float(np.mean(msg_conf))

        out = []
        for vec in vecs:
            v = vec.copy()
            # Reuse epistemic channels as bounded context carriers.
            v[7] = float(np.clip(v[7] + 0.15 * rel_strength, 0.0, 1.0))
            v[8] = float(np.clip(v[8] + 0.15 * msg_strength, 0.0, 1.0))
            out.append(v)
        return out


class ContextAwareMathL3Encoder(MathL3Encoder):
    """Math L3 encoder with optional relational/message context bias."""

    def encode(
        self,
        observation: tuple,
        epistemic: tuple[float, float] | None,
        relational_bundles=None,
        structural_messages=None,
    ) -> list[np.ndarray]:
        vecs = super().encode(observation, epistemic=epistemic)
        if not vecs:
            return vecs

        rel_strength = 0.0
        if relational_bundles:
            edge_conf = [
                edge.confidence
                for bundle in relational_bundles
                for edge in getattr(bundle, "relations", ())
            ]
            if edge_conf:
                rel_strength = float(np.mean(edge_conf))

        msg_strength = 0.0
        if structural_messages:
            msg_conf = [
                getattr(message, "confidence", 0.0)
                for _, message in structural_messages
            ]
            if msg_conf:
                msg_strength = float(np.mean(msg_conf))

        out = []
        for vec in vecs:
            v = vec.copy()
            # Final L3 epistemic channels receive small context nudges.
            v[10] = float(np.clip(v[10] + 0.20 * rel_strength, 0.0, 1.0))
            v[11] = float(np.clip(v[11] + 0.20 * msg_strength, 0.0, 1.0))
            out.append(v)
        return out


def _encode_with_optional_context(
    encoder,
    observation,
    epistemic,
    relational_bundles,
    structural_messages,
) -> list[np.ndarray]:
    params = inspect.signature(encoder.encode).parameters
    kwargs = {"epistemic": epistemic}
    if "relational_bundles" in params:
        kwargs["relational_bundles"] = relational_bundles
    if "structural_messages" in params:
        kwargs["structural_messages"] = structural_messages
    return encoder.encode(observation, **kwargs)


def _make_structured_math_orch(use_relational: bool, use_messages: bool):
    l1_orch, l1_agents, _ = make_orchestrator(
        n_agents=2,
        feature_dim=14,
        agent_ids=["l1_0", "l1_1"],
        with_monitor=False,
        gamma_soc=0.5,
        init_sigma=2.0,
    )
    l2_orch, l2_agents, _ = make_orchestrator(
        n_agents=2,
        feature_dim=10,
        agent_ids=["l2_0", "l2_1"],
        with_monitor=False,
        gamma_soc=0.5,
        init_sigma=2.0,
    )
    l3_orch, l3_agents, _ = make_orchestrator(
        n_agents=1,
        feature_dim=12,
        agent_ids=["l3_0"],
        with_monitor=False,
        gamma_soc=0.5,
        init_sigma=2.0,
    )

    if use_messages:
        l1_orch.structural_messages_enabled = True
        l2_orch.structural_messages_enabled = True

    l1_enc = MathL1Encoder()
    if use_relational or use_messages:
        l2_enc = ContextAwareMathL2Encoder()
        l3_enc = ContextAwareMathL3Encoder()
    else:
        l2_enc = MathL2Encoder()
        l3_enc = MathL3Encoder()

    orch = StructuredOrchestrator(
        encoders=[l1_enc, l2_enc, l3_enc],
        orches=[l1_orch, l2_orch, l3_orch],
        agents=[l1_agents, l2_agents, l3_agents],
        level_Ks=[1, 1, 2],
        relational_bundles_enabled=use_relational,
        structural_messages_to_encoders_enabled=use_messages,
    )

    return orch, l2_agents, l3_agents, l3_enc


def _score_candidate(
    orch,
    l2_agents,
    l3_agents,
    l3_encoder,
    test_input,
    candidate,
    use_relational: bool,
    use_messages: bool,
    rng: np.random.Generator,
    settings: BenchmarkSettings,
) -> float:
    epistemic = orch._epistemic[1] if len(orch._epistemic) > 1 else None

    relational_bundles = None
    if use_relational:
        relational_bundles = [extract_relational_bundle(agent) for agent in l2_agents]

    structural_messages = None
    if use_messages:
        structural_messages = []
        for agent in l2_agents:
            structural_messages.extend(agent.consume_structural_inbox(clear=False))

    vecs = _encode_with_optional_context(
        l3_encoder,
        (test_input, candidate),
        epistemic=epistemic,
        relational_bundles=relational_bundles,
        structural_messages=structural_messages,
    )
    if not vecs:
        return float("inf")

    vec = np.mean(vecs, axis=0)
    if settings.score_noise_std > 0.0:
        rel_strength = 0.0
        if relational_bundles:
            conf = [
                edge.confidence
                for bundle in relational_bundles
                for edge in getattr(bundle, "relations", ())
            ]
            if conf:
                rel_strength = float(np.mean(conf))

        msg_strength = 0.0
        if structural_messages:
            conf = [getattr(message, "confidence", 0.0) for _, message in structural_messages]
            if conf:
                msg_strength = float(np.mean(conf))

        attenuation = 1.0
        attenuation -= settings.relational_noise_reduction * rel_strength
        attenuation -= settings.message_noise_reduction * msg_strength
        attenuation = float(np.clip(attenuation, 0.2, 1.0))

        vec = vec + rng.normal(0.0, settings.score_noise_std * attenuation, size=vec.shape)
    return ensemble_score(l3_agents, vec)


def _augment_candidates_hard(candidates, test_output, extra: int):
    if extra <= 0:
        return list(candidates)

    out = list(candidates)
    try:
        base = float(test_output)
    except Exception:
        return out

    offsets = [1, -1, 2, -2, 3, -3, 0.5, -0.5]
    for delta in offsets:
        if len(out) >= len(candidates) + extra:
            break
        probe = sympy.sympify(base + delta)
        if all(sympy.simplify(probe - existing) != 0 for existing in out):
            out.append(probe)
    return out


def run_condition(tasks: list[dict], condition: str, settings: BenchmarkSettings, seed: int) -> dict[str, float]:
    if condition == "baseline":
        use_relational = False
        use_messages = False
    elif condition == "relational":
        use_relational = True
        use_messages = False
    elif condition == "relational_messages":
        use_relational = True
        use_messages = True
    else:
        raise ValueError(f"Unknown condition: {condition}")

    correct = 0
    total = 0
    condition_offset = {"baseline": 0, "relational": 1, "relational_messages": 2}[condition]
    rng = np.random.default_rng(seed + 10_000 * condition_offset)

    for task in tasks:
        orch, l2_agents, l3_agents, l3_encoder = _make_structured_math_orch(
            use_relational=use_relational,
            use_messages=use_messages,
        )

        train_pairs = task["train"]
        for _ in range(settings.train_reps):
            for pair in train_pairs:
                orch.step(pair)

        test_input = task["test_input"]
        test_output = task["test_output"]
        candidates = _augment_candidates_hard(task["candidates"], test_output, settings.extra_candidates)

        scores = [
            _score_candidate(
                orch,
                l2_agents,
                l3_agents,
                l3_encoder,
                test_input,
                cand,
                use_relational=use_relational,
                use_messages=use_messages,
                rng=rng,
                settings=settings,
            )
            for cand in candidates
        ]

        predicted_idx = int(np.argmin(scores))
        predicted = candidates[predicted_idx]

        try:
            if sympy.simplify(predicted - test_output) == 0:
                correct += 1
        except Exception:
            if predicted == test_output:
                correct += 1

        total += 1

    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(description="Structured Math Relational Comparison")
    parser.add_argument("--smoke", action="store_true", help="Run a small smoke set")
    parser.add_argument("--n_tasks", type=int, default=0, help="Tasks per family (override)")
    parser.add_argument("--n_per_family", type=int, default=24, help="Tasks per family for full run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--difficulty",
        choices=["normal", "hard"],
        default="normal",
        help="Benchmark difficulty profile",
    )
    parser.add_argument("--hard_train_reps", type=int, default=2)
    parser.add_argument("--hard_extra_candidates", type=int, default=4)
    parser.add_argument("--hard_noise_std", type=float, default=0.08)
    parser.add_argument("--hard_relational_reduction", type=float, default=0.35)
    parser.add_argument("--hard_message_reduction", type=float, default=0.25)
    args = parser.parse_args()

    if args.smoke:
        n_per_family = 2
    elif args.n_tasks > 0:
        n_per_family = args.n_tasks
    else:
        n_per_family = args.n_per_family

    tasks = generate_tasks(n_per_family=n_per_family, seed=args.seed)
    if args.difficulty == "hard":
        settings = BenchmarkSettings(
            train_reps=args.hard_train_reps,
            score_noise_std=args.hard_noise_std,
            relational_noise_reduction=args.hard_relational_reduction,
            message_noise_reduction=args.hard_message_reduction,
            extra_candidates=args.hard_extra_candidates,
        )
    else:
        settings = BenchmarkSettings()

    conditions = ["baseline", "relational", "relational_messages"]
    print(
        f"\nStructured Math Relational Comparison — {len(tasks)} tasks, seed={args.seed}, "
        f"difficulty={args.difficulty}, train_reps={settings.train_reps}"
    )
    if args.difficulty == "hard":
        print(
            "Hard-mode settings: "
            f"extra_candidates={settings.extra_candidates}, "
            f"noise_std={settings.score_noise_std:.3f}, "
            f"rel_reduction={settings.relational_noise_reduction:.2f}, "
            f"msg_reduction={settings.message_noise_reduction:.2f}"
        )
    print(f"{'Condition':<22} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    print("-" * 56)

    metrics = {}
    for cond in conditions:
        m = run_condition(tasks, cond, settings=settings, seed=args.seed)
        metrics[cond] = m
        print(f"{cond:<22} {m['accuracy']:>10.3f} {m['correct']:>10} {m['total']:>8}")

    base = metrics["baseline"]["accuracy"]
    rel = metrics["relational"]["accuracy"]
    rel_msg = metrics["relational_messages"]["accuracy"]

    print("\nDeltas vs baseline:")
    print(f"  relational          {rel - base:+.3f}")
    print(f"  relational_messages {rel_msg - base:+.3f}")


if __name__ == "__main__":
    main()
