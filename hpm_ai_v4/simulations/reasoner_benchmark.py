"""Domain-agnostic benchmark for the generic HPM reasoner."""
import argparse
import json
import os
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional

import numpy as np

from hpm_ai_v4.agents.reasoning import HypothesisFrame
from hpm_ai_v4.io.adapters import CodeDSLAdapter
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary
from hpm_ai_v4.tools.text_signals import TextSignalExtractor


@dataclass
class BenchmarkCase:
    family: str
    expected_mode: str
    feature_pack: Dict[str, Any]
    seed_text: str = ""
    target_text: str = ""
    corrupted_text: str = ""
    target_sequence: Optional[List[int]] = None
    control_context: Optional[List[int]] = None


def _corrupt_text(text: str) -> str:
    out: List[str] = []
    for idx, ch in enumerate(text):
        if ch.isalpha() and idx % 13 == 0:
            continue
        if ch.isspace() and idx % 11 == 0:
            continue
        if ch.isdigit() and idx % 7 == 0:
            continue
        out.append(ch)
    return "".join(out)


def _score_margin(frames: List[HypothesisFrame]) -> float:
    if len(frames) < 2:
        return float(frames[0].score if frames else 0.0)
    return float(frames[0].score - frames[1].score)


def _mean_metric(history: List[Dict[str, Any]], key: str) -> float:
    vals = [float(s.get(key, 0.0)) for s in history if key in s]
    return float(mean(vals)) if vals else 0.0


def _mean_nested_metric(history: List[Dict[str, Any]], outer: str, inner: str) -> float:
    vals: List[float] = []
    for snap in history:
        nested = snap.get(outer, {})
        if isinstance(nested, dict) and inner in nested:
            vals.append(float(nested.get(inner, 0.0)))
    return float(mean(vals)) if vals else 0.0


def _benchmark_report(history: List[Dict[str, Any]]) -> None:
    if not history:
        print("No benchmark metrics recorded.")
        return

    print("\n" + "=" * 72)
    print("HPM REASONER BENCHMARK REPORT")
    print("=" * 72)
    print(
        f"task_score={_mean_metric(history, 'task_score'):.3f} "
        f"mode_accuracy={_mean_metric(history, 'mode_accuracy'):.3f} "
        f"hypothesis_margin={_mean_metric(history, 'hypothesis_margin'):.3f} "
        f"graph_support={_mean_metric(history, 'graph_support'):.3f}"
    )
    print(
        f"L1_mi={_mean_nested_metric(history, 'pattern', 'l1_mi'):.3f} "
        f"L2_mi={_mean_nested_metric(history, 'pattern', 'l2_mi'):.3f} "
        f"L3_mi={_mean_nested_metric(history, 'pattern', 'l3_mi'):.3f} "
        f"L4_mi={_mean_nested_metric(history, 'pattern', 'l4_mi'):.3f} "
        f"L5_mi={_mean_nested_metric(history, 'pattern', 'l5_mi'):.3f}"
    )
    for phase in ("adapt", "eval"):
        phase_hist = [snap for snap in history if snap.get("phase") == phase]
        if phase_hist:
            print(
                f"{phase}_task={_mean_metric(phase_hist, 'task_score'):.3f} "
                f"{phase}_mode_acc={_mean_metric(phase_hist, 'mode_accuracy'):.3f} "
                f"{phase}_margin={_mean_metric(phase_hist, 'hypothesis_margin'):.3f}"
            )
    print("=" * 72)


def _pattern_snapshot(layered: LayeredAgent) -> Dict[str, Any]:
    recent_l1 = list(layered._l1_state_history[-200:])
    recent_l2 = list(layered._l2_state_history[-200:])
    l1_metrics = layered.l1_metrics()
    l2_metrics = layered.l2_metrics(recent_l1)
    l3_metrics = layered.l3_metrics(recent_l2)
    return {
        "l1_mi": l1_metrics["mi"],
        "l2_mi": l2_metrics["mi"],
        "l2_accuracy": l2_metrics["accuracy"],
        "l3_mi": l3_metrics["mi"],
        "l3_accuracy": l3_metrics["accuracy"],
        "l4_mi": layered.l4_metrics()["mi"],
        "l5_mi": layered.l5_metrics()["mi"],
    }


def _run_case(
    layered: LayeredAgent,
    case: BenchmarkCase,
    text_signals: TextSignalExtractor,
    code_adapter: CodeDSLAdapter,
) -> Dict[str, Any]:
    feature_pack = dict(case.feature_pack)
    control_snapshot = layered.l1.reasoner.control_context(feature_pack=feature_pack)
    learned_mode_prior = control_snapshot.get("mode_prior", {})
    if learned_mode_prior:
        feature_pack.setdefault("mode_prior", learned_mode_prior)
    if case.family == "continue":
        target_sequence = [layered._adapter.encode_char(ch) for ch in case.target_text if 32 <= ord(ch) <= 126 or ch == "\n"]
        frames = layered.l1.reasoner.plan_hypotheses(
            goal_state=layered._class_name_to_id("space"),
            target_sequence=target_sequence,
            feature_pack=feature_pack,
        )
        planned_text = layered.plan_text_continuation(
            target_text=case.target_text,
            seed_text=case.seed_text,
            horizon=max(8, len(case.target_text) // 2),
            feature_pack=feature_pack,
        )
        generated_text = layered.generate_text(
            steps=max(8, len(case.target_text) // 2),
            seed_text=case.seed_text,
            target_text=case.target_text,
            mode="target",
            include_seed=False,
            update_policy=False,
            context_features=feature_pack,
        )
        target_stats = layered.evaluate_generated_text(generated_text, case.target_text)
        planned_stats = layered.evaluate_generated_text(planned_text, case.target_text)
        replay_signal = text_signals.analyze(
            generated_text,
            context_texts=[case.seed_text, case.target_text],
            target_text=case.target_text,
            dictionary=layered.dictionary,
            grammar=layered.grammar,
        )
        layered.observe_text(case.target_text, feedback_mode="hybrid", generated_text=generated_text, self_feedback_weight=0.02, feedback_signal=replay_signal.to_dict())
    elif case.family == "repair":
        target_sequence = [layered._adapter.encode_char(ch) for ch in case.target_text if 32 <= ord(ch) <= 126 or ch == "\n"]
        frames = layered.l1.reasoner.plan_hypotheses(
            goal_state=layered._class_name_to_id("space"),
            target_sequence=target_sequence,
            feature_pack=feature_pack,
        )
        planned_text = layered.plan_text_continuation(
            target_text=case.target_text,
            seed_text=case.corrupted_text,
            horizon=max(8, len(case.target_text) // 2),
            feature_pack=feature_pack,
        )
        generated_text = layered.repair_text(
            corrupted_text=case.corrupted_text,
            target_text=case.target_text,
            mode="target",
            update_policy=True,
        )
        target_stats = layered.evaluate_generated_text(generated_text, case.target_text)
        planned_stats = layered.evaluate_generated_text(planned_text, case.target_text)
        repair_signal = text_signals.analyze(
            generated_text,
            context_texts=[case.corrupted_text, case.target_text],
            target_text=case.target_text,
            dictionary=layered.dictionary,
            grammar=layered.grammar,
        )
        layered.observe_text(case.target_text, feedback_mode="hybrid", generated_text=generated_text, self_feedback_weight=0.02, feedback_signal=repair_signal.to_dict())
    elif case.family == "code_dsl":
        target_sequence = [layered._adapter.encode_char(ch) for ch in case.target_text if 32 <= ord(ch) <= 126 or ch == "\n"]
        frames = layered.l1.reasoner.plan_hypotheses(
            goal_state=None,
            target_sequence=target_sequence,
            feature_pack=feature_pack,
        )
        planned_text = layered.repair_text(
            corrupted_text=case.corrupted_text,
            target_text=case.target_text,
            use_constraints=True,
            mode="target",
            update_policy=True,
        )
        generated_text = planned_text
        target_stats = layered.evaluate_generated_text(generated_text, case.target_text)
        planned_stats = target_stats
        signal_pack = text_signals.analyze(
            generated_text,
            context_texts=[case.corrupted_text, case.target_text],
            target_text=case.target_text,
            dictionary=layered.dictionary,
            grammar=layered.grammar,
        )
        layered.observe_code_dsl(
            case.target_text,
            generated_program=generated_text,
            feedback_mode="hybrid",
            self_feedback_weight=0.02,
            feedback_signal=signal_pack.to_dict(),
        )
        target_stats["parseable"] = bool(code_adapter.from_text(generated_text))
        target_value = code_adapter.execute(case.target_text)
        generated_value = code_adapter.execute(generated_text)
        target_stats["execution_match"] = generated_value == target_value and generated_value is not None
    else:
        if case.control_context:
            for obs in case.control_context:
                layered.perceive(obs)
        frames = layered.l1.reasoner.plan_hypotheses(
            goal_state=case.target_sequence[-1] if case.target_sequence else None,
            target_sequence=case.target_sequence,
            feature_pack=feature_pack,
        )
        planned_text = "".join(str(x) for x in frames[0].sequence) if frames else ""
        generated_text = planned_text
        target_stats = {
            "token_agreement": float(sum(1 for a, b in zip(frames[0].sequence if frames else [], case.target_sequence or []) if a == b) / max(1, len(case.target_sequence or []))),
            "plausibility": 0.0,
        }
        planned_stats = target_stats

    control_ctx = layered.l1.reasoner.control_context(layered._raw_history, feature_pack=case.feature_pack)
    poly_summary = layered.l1.reasoner.polygraph.projection_summary(
        layered._raw_history,
        query_action=None,
        query_stage=control_ctx.get("dominant_stage"),
        query_policy=control_ctx.get("dominant_policy"),
    )

    if frames:
        top_frame = frames[0]
        mode_accuracy = 1.0 if top_frame.mode == case.expected_mode else 0.0
        hypothesis_margin = _score_margin(frames)
        selected_mode = top_frame.mode
        selected_score = top_frame.score
        memory_hits = len(top_frame.memory_hits or [])
    else:
        mode_accuracy = 0.0
        hypothesis_margin = 0.0
        selected_mode = "unknown"
        selected_score = 0.0
        memory_hits = 0

    pattern = _pattern_snapshot(layered)
    return {
        "family": case.family,
        "expected_mode": case.expected_mode,
        "selected_mode": selected_mode,
        "mode_accuracy": mode_accuracy,
        "selected_score": selected_score,
        "hypothesis_margin": hypothesis_margin,
        "task_score": float(
            0.7 * float(target_stats.get("token_agreement", 0.0))
            + 0.3 * float(target_stats.get("execution_match", False))
            if case.family == "code_dsl"
            else float(target_stats.get("token_agreement", 0.0))
        ),
        "planned_score": float(planned_stats.get("token_agreement", 0.0)),
        "plausibility": float(target_stats.get("plausibility", 0.0)),
        "graph_support": float(poly_summary.get("multi_supported_count", 0)),
        "community_strength": float(control_ctx.get("community_strength", 0.0)),
        "memory_hits": memory_hits,
        "graph_summary": poly_summary,
        "control_context": control_ctx,
        "mode_prior": dict(feature_pack.get("mode_prior", {})),
        "generated_text": generated_text[:160],
        "planned_text": planned_text[:160],
        "pattern": pattern,
    }


def run_reasoner_benchmark(
    checkpoint_dir: str = ".",
    num_workers: int = 1,
    use_dict: bool = True,
    library_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run a generic reasoner benchmark across multiple task families."""
    np.random.seed(7)
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    layered = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)
    text_signals = TextSignalExtractor()
    code_adapter = CodeDSLAdapter()

    if library_path and os.path.exists(library_path + ".l1.pkl"):
        layered.load_bundle(library_path)

    warmup_text = "the system learns from context and memory.\n"
    layered.observe_text(warmup_text, feedback_mode="target")

    continue_case = BenchmarkCase(
        family="continue",
        expected_mode="continue",
        seed_text="the system learns from context and",
        target_text="the system learns from context and continues with structure.",
        feature_pack={
            "desired_mode": "continue",
            "mode_prior": {"continue": 1.0},
            "target_strength": 0.7,
            "diversity_weight": 0.1,
        },
    )
    repair_case = BenchmarkCase(
        family="repair",
        expected_mode="repair",
        seed_text="repair me",
        target_text="structured memory helps the system recover missing words.",
        corrupted_text=_corrupt_text("structured memory helps the system recover missing words."),
        feature_pack={
            "target_strength": 1.0,
            "constraint_strength": 0.2,
            "corruption_strength": 1.0,
        },
    )
    code_program = "push 2\npush 3\nadd\npush 4\nmul\nreturn"
    code_case = BenchmarkCase(
        family="code_dsl",
        expected_mode="constrain",
        target_text=code_adapter.to_text(code_program),
        corrupted_text=_corrupt_text(code_adapter.to_text(code_program)),
        target_sequence=code_adapter.to_observations(code_program, max_length=32),
        feature_pack={
            "constraint_strength": 1.0,
            "target_strength": 0.9,
            "corruption_strength": 1.0,
        },
    )
    control_case = BenchmarkCase(
        family="control",
        expected_mode="explore",
        target_sequence=[1, 0, 1, 0],
        control_context=[0, 1, 0, 1, 1, 0],
        feature_pack={
            "diversity_weight": 1.0,
            "target_strength": 0.2,
        },
    )

    cases = [continue_case, repair_case, code_case, control_case]
    history: List[Dict[str, Any]] = []
    for phase, learn in (("adapt", True), ("eval", False)):
        for case in cases:
            result = _run_case(
                layered,
                case,
                text_signals,
                code_adapter,
            )
            result["phase"] = phase
            history.append(result)
            if learn:
                outcome_score = 0.7 * float(result.get("task_score", 0.0)) + 0.3 * float(result.get("mode_accuracy", 0.0))
                layered.l1.reasoner.observe_outcome(
                    actual_obs=layered._adapter.encode_char(" "),
                    context_obs=layered._raw_history,
                    metadata={
                        "mode": result["selected_mode"],
                        "task_family": result["family"],
                        "phase": phase,
                    },
                    feature_pack={
                        "token_agreement": float(result.get("task_score", 0.0)),
                        "plausibility": float(result.get("plausibility", 0.0)),
                        "structural_score": float(result.get("pattern", {}).get("l4_mi", 0.0)),
                    },
                )

    os.makedirs(checkpoint_dir, exist_ok=True)
    base = os.path.join(checkpoint_dir, "reasoner_benchmark_library")
    layered.save_bundle(base)
    with open(base + ".mode_prior.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "learned_mode_prior": layered.l1.reasoner.control_context().get("learned_mode_prior", {}),
                "mode_prior": layered.l1.reasoner.control_context().get("mode_prior", {}),
                "mode_selection_counts": layered.l1.reasoner.control_context().get("mode_selection_counts", {}),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    _benchmark_report(history)
    return history


def _parse_args():
    p = argparse.ArgumentParser(description="Domain-agnostic HPM reasoner benchmark")
    p.add_argument("--checkpoint-dir", default=".", help="Directory for checkpoint files")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dict", action="store_true", help="Enable dictionary and grammar validators")
    p.add_argument("--library", default=None, help="Base path to pre-built pattern library (no .pkl suffix)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_reasoner_benchmark(
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.workers,
        use_dict=args.dict,
        library_path=args.library,
    )
