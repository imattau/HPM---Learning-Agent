"""Audit the current first-target training gates from saved experiment reports.

The audit keeps task accuracy, answer quality, correction safety, user ratings,
retention, and latency separate. A missing report is reported as unmeasured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected")
DEFAULT_COMPARISON = ROOT / "full405_9b_comparison.json"
DEFAULT_HPM = ROOT / "balanced_six_family_report.json"
DEFAULT_QWEN_RAW = ROOT / "qwen_bm25_full405.json"
DEFAULT_CORRECTION = ROOT / "correction-transfer-v4-r6/cumulative_study_report.json"
DEFAULT_ACQUISITION = Path(
    "hpm_ai_v2/experiments/results/2026-10-24-knowledge-acquisition-growth-150-current/"
    "cumulative_study_report.json"
)


def _read(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _gate(status: str, **evidence: Any) -> dict[str, Any]:
    return {"status": status, **evidence}


def audit_training_acceptance(
    comparison_path: str | Path = DEFAULT_COMPARISON,
    hpm_path: str | Path = DEFAULT_HPM,
    qwen_raw_path: str | Path = DEFAULT_QWEN_RAW,
    correction_path: str | Path = DEFAULT_CORRECTION,
    usefulness_path: str | Path | None = None,
    acquisition_path: str | Path | None = None,
) -> dict[str, Any]:
    comparison = _read(comparison_path)
    hpm_report = _read(hpm_path)
    qwen_raw = _read(qwen_raw_path)
    correction = _read(correction_path)
    usefulness = _read(usefulness_path)
    acquisition = _read(acquisition_path)
    if comparison is None or hpm_report is None:
        raise FileNotFoundError("comparison and HPM reports are required")

    hpm, qwen = comparison["hpm"], comparison["qwen_bm25"]
    # Cumulative-study reports use retention_after_all_phases; the earlier
    # balanced-six-family report used final_exam. Accept both without
    # reshaping or dropping episode-level evidence.
    hpm_final = hpm_report.get(
        "final_exam", hpm_report.get("retention_after_all_phases", {})
    )
    family = hpm.get("per_task_type", {})
    family_deltas = {
        task: {
            "hpm_accuracy": metrics.get("accuracy"),
            "baseline_accuracy": qwen.get("per_task_type", {}).get(task, {}).get("accuracy"),
            "difference": (
                metrics.get("accuracy", 0.0)
                - qwen.get("per_task_type", {}).get(task, {}).get("accuracy", 0.0)
            ),
        }
        for task, metrics in family.items()
    }
    minimum_family_count = min((metric.get("total", 0) for metric in family.values()), default=0)
    episode_count = int(comparison.get("episode_count", hpm.get("total", 0)))
    family_coverage = episode_count >= 300 and minimum_family_count >= 60
    overall_delta = hpm["accuracy"] - qwen["accuracy"]
    family_noninferior = all(
        row["difference"] >= -0.10 for row in family_deltas.values()
    )
    task_gate = episode_count >= 300 and overall_delta >= -0.05 and family_noninferior

    accuracy_comparison = {
        "hpm_grounded_answer_accuracy": hpm.get("grounded_answer_accuracy"),
        "baseline_grounded_answer_accuracy": qwen.get("grounded_answer_accuracy"),
        "grounded_answer_accuracy_difference": (
            hpm.get("grounded_answer_accuracy", 0.0)
            - qwen.get("grounded_answer_accuracy", 0.0)
        ),
        "hpm_answer_token_f1": hpm.get("answer_token_f1"),
        "baseline_answer_token_f1": qwen.get("answer_token_f1"),
        "answer_token_f1_difference": hpm.get("answer_token_f1", 0.0)
        - qwen.get("answer_token_f1", 0.0),
        "hpm_evidence_accuracy": hpm.get("evidence_accuracy"),
        "baseline_evidence_accuracy": qwen.get("evidence_accuracy"),
    }
    accuracy_pass = (
        accuracy_comparison["grounded_answer_accuracy_difference"] >= 0
        and accuracy_comparison["answer_token_f1_difference"] >= 0
    )
    factual_status = "pass" if accuracy_pass else (
        "mixed" if accuracy_comparison["grounded_answer_accuracy_difference"] >= 0
        else "fail"
    )

    unsupported = {"hpm": None, "baseline": None}
    if qwen_raw is not None:
        qwen_rows = qwen_raw.get("results", {}).get("episodes", [])
        abstention_rows = [row for row in qwen_rows if row.get("task_type") == "abstain"]
        if abstention_rows:
            unsupported["baseline"] = sum(bool(row.get("abstained")) for row in abstention_rows) / len(abstention_rows)
    hpm_rows = hpm_final.get("episode_results", [])
    hpm_abstention_rows = [row for row in hpm_rows if row.get("task_type") == "abstain"]
    if hpm_abstention_rows:
        unsupported["hpm"] = sum(bool(row.get("abstained")) for row in hpm_abstention_rows) / len(hpm_abstention_rows)
    unsupported_status = (
        "unmeasured" if None in unsupported.values()
        else "pass" if unsupported["hpm"] >= unsupported["baseline"]
        else "fail"
    )

    retention_passed = hpm_report.get("all_retention_gates_passed")
    if retention_passed is None:
        retention_passed = hpm_report.get("training", {}).get(
            "all_retention_gates_passed"
        )
    retention_status = "unmeasured" if retention_passed is None else ("pass" if retention_passed else "fail")

    correction_gate: dict[str, Any]
    if correction is None:
        correction_gate = _gate("unmeasured")
    else:
        phases = correction.get("training", {}).get("phases", [])
        phase = phases[-1] if phases else {}
        after = phase.get("phase_test_metrics", {})
        before = phase.get("phase_test_before_metrics", {})
        correction_type = after.get("per_task_type", {}).get("correction", {})
        correction_accuracy = correction_type.get("accuracy")
        validation_gate = phase.get("fit", {}).get("fact_validation_gate", {})
        validation_passed = validation_gate.get("passed")
        controls_before = {
            row["episode_id"]: row
            for row in before.get("episode_results", [])
            if "collateral" in row.get("episode_id", "")
        }
        controls_after = {
            row["episode_id"]: row
            for row in after.get("episode_results", [])
            if "collateral" in row.get("episode_id", "")
        }
        paired = controls_before.keys() & controls_after.keys()
        regressions = [
            case_id for case_id in paired
            if controls_after[case_id].get("answer_token_f1", 0.0)
            < controls_before[case_id].get("answer_token_f1", 0.0)
        ]
        changed_outputs = [
            case_id for case_id in paired
            if any(
                controls_before[case_id].get(field) != controls_after[case_id].get(field)
                for field in ("answer", "selected_evidence_id")
            )
        ]
        correction_ok = correction_accuracy is not None and correction_accuracy >= 0.80
        safety_ok = bool(paired) and not regressions and not changed_outputs
        correction_gate = _gate(
            "pass" if correction_ok and safety_ok and validation_passed is True else "fail",
            correction_accuracy=correction_accuracy,
            correction_cases=correction_type.get("total"),
            fact_validation_gate_passed=validation_passed,
            fact_validation_validated=validation_gate.get("validated_training_fact_count"),
            fact_validation_total=validation_gate.get("training_fact_count"),
            collateral_cases=len(paired),
            collateral_answer_quality_regressions=regressions,
            collateral_outputs_changed=changed_outputs,
            correction_matching=correction.get("configuration", {}).get("correction_matching"),
        )

    if usefulness is None:
        usefulness_gate = _gate("unmeasured", reason="No completed rating analysis was supplied")
    else:
        gate = usefulness.get("user_acceptance_gate", {})
        passed = gate.get("passed")
        usefulness_gate = _gate("unmeasured" if passed is None else ("pass" if passed else "fail"), **gate)

    if acquisition is None:
        acquisition_gate = _gate("unmeasured", reason="No acquisition report was supplied")
        transfer_effect_gate = _gate("unmeasured", reason="No acquisition report was supplied")
    else:
        acquisition_phases = acquisition.get("training", {}).get("phases", [])
        # Cumulative runs may append later abstention/workflow phases after
        # the knowledge acquisition phase. Audit the latest phase that trained
        # facts, not merely the final phase in the report.
        acquisition_phase = next(
            (
                phase for phase in reversed(acquisition_phases)
                if phase.get("fit", {}).get("fact_validation_gate", {}).get(
                    "training_fact_count", 0
                )
            ),
            acquisition_phases[-1] if acquisition_phases else {},
        )
        acquisition_fit = acquisition_phase.get("fit", {})
        acquisition_fact_gate = acquisition_fit.get("fact_validation_gate", {})
        acquisition_growth = acquisition_fit.get("knowledge_growth", {})
        acquisition_before = acquisition_phase.get("phase_test_before_metrics", {})
        acquisition_after = acquisition_phase.get("phase_test_metrics", {})
        transfer_delta = acquisition_phase.get("phase_test_accuracy_delta")
        acquisition_restart = acquisition.get("restart_consistency", {}).get("same_predictions")
        checkpoint_path = acquisition.get("checkpoint")
        checkpoint_file = Path(checkpoint_path) if checkpoint_path else None
        if checkpoint_file is not None and checkpoint_file.is_dir():
            checkpoint_file = checkpoint_file / "study_training_state.json"
        checkpoint_state = (
            _read(checkpoint_file) if checkpoint_file else None
        )
        persisted_fact_data = (checkpoint_state or {}).get("fact_node_data", {})
        persisted_fact_statuses = [row.get("status") for row in persisted_fact_data.values()]
        persisted_validated_facts = sum(status in {"validated", "promoted"} for status in persisted_fact_statuses)
        new_facts = acquisition_growth.get("new_fact_nodes", 0)
        validated_training_facts = acquisition_fact_gate.get("validated_training_fact_count") or 0
        acquisition_passed = (
            acquisition_fact_gate.get("passed") is True
            and new_facts > 0
            and validated_training_facts > 0
            and persisted_validated_facts >= validated_training_facts
            and acquisition_restart is True
        )
        acquisition_gate = _gate(
            "pass" if acquisition_passed else "fail",
            new_fact_nodes=new_facts,
            validated_training_facts=validated_training_facts,
            training_facts=acquisition_fact_gate.get("training_fact_count"),
            fact_validation_passed=acquisition_fact_gate.get("passed"),
            checkpoint_fact_nodes_persisted=len(persisted_fact_data),
            checkpoint_validated_fact_nodes=persisted_validated_facts,
            restart_predictions_match=acquisition_restart,
            test_accuracy_before=acquisition_before.get("accuracy"),
            test_accuracy_after=acquisition_after.get("accuracy"),
            test_accuracy_delta=acquisition_phase.get("phase_test_accuracy_delta"),
            all_retention_gates_passed=acquisition.get("training", {}).get("all_retention_gates_passed"),
        )
        transfer_status = (
            "unmeasured" if transfer_delta is None else
            "pass" if transfer_delta > 0 else
            "fail" if transfer_delta < 0 else
            "mixed"
        )
        transfer_effect_gate = _gate(
            transfer_status,
            accuracy_before=acquisition_before.get("accuracy"),
            accuracy_after=acquisition_after.get("accuracy"),
            accuracy_delta=transfer_delta,
            interpretation=(
                "held-out performance improved" if transfer_delta is not None and transfer_delta > 0 else
                "held-out performance regressed" if transfer_delta is not None and transfer_delta < 0 else
                "no measured held-out accuracy gain; test may be saturated"
            ),
        )

    protocol = comparison.get("protocol", {})
    baseline_configuration = (
        qwen_raw.get("results", {}).get("configuration", {})
        if qwen_raw is not None else {}
    )
    latency_status = (
        "unmeasured" if hpm.get("median_latency_ms") is None or qwen.get("median_latency_ms") is None
        else "mixed"  # response time is measured; user effort is not
    )
    return {
        "benchmark": comparison.get("benchmark"),
        "gates": {
            "episode_and_family_coverage": _gate(
                "pass" if family_coverage else "fail",
                episode_count=episode_count,
                minimum_family_count=minimum_family_count,
                family_counts={key: val.get("total", 0) for key, val in family.items()},
            ),
            "task_completion_vs_9b": _gate(
                "pass" if task_gate else "fail",
                overall_difference=overall_delta,
                maximum_allowed_overall_shortfall=0.05,
                maximum_allowed_family_shortfall=0.10,
                per_family=family_deltas,
            ),
            "grounded_answer_quality": _gate(factual_status, **accuracy_comparison),
            "unsupported_answer_rejection": _gate(unsupported_status, **unsupported),
            "correction_transfer_and_safety": correction_gate,
            "validated_knowledge_growth": acquisition_gate,
            "knowledge_transfer_effect": transfer_effect_gate,
            "retention": _gate(retention_status, all_phase_gates_passed=retention_passed),
            "blinded_human_usefulness": usefulness_gate,
            "latency": _gate(
                latency_status,
                hpm_median_ms=hpm.get("median_latency_ms"),
                baseline_median_ms=qwen.get("median_latency_ms"),
                user_effort="unmeasured",
            ),
        },
        "baseline_configuration": {
            "model": protocol.get("ollama_model"),
            "digest": protocol.get("ollama_digest"),
            "parameter_size": protocol.get("parameter_size"),
            "quantization": protocol.get("quantization"),
            "run_configuration": baseline_configuration,
            "shared_sources": protocol.get("shared_sources"),
            "shared_train_validation_interactions": protocol.get("shared_prior_train_validation_interactions"),
        },
        "hpm_configuration": {
            "system": "HPM AI v2 supervised text study",
            "encoder": "TF-IDF text domain",
            "answer_generation": "extractive source sentence selection; no language generator",
            "tool_access": "local source corpus and learned HFN state",
        },
        "acceptance_passed": all(
            value["status"] == "pass"
            for value in [
                {"status": "pass" if family_coverage else "fail"},
                {"status": "pass" if task_gate else "fail"},
                {"status": factual_status},
                {"status": unsupported_status},
                correction_gate,
                acquisition_gate,
                transfer_effect_gate,
                {"status": retention_status},
                usefulness_gate,
                {"status": latency_status},
            ]
        ),
        "limitations": [
            "The benchmark is hand-authored and uses a shared 38-source corpus.",
            "The training history and tasks model one benchmark user; broad user generalization is unmeasured.",
            "A missing human-rating report leaves the usefulness gate unmeasured.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--hpm-report", type=Path, default=DEFAULT_HPM)
    parser.add_argument("--qwen-raw", type=Path, default=DEFAULT_QWEN_RAW)
    parser.add_argument("--correction-report", type=Path, default=DEFAULT_CORRECTION)
    parser.add_argument("--acquisition-report", type=Path, default=DEFAULT_ACQUISITION)
    parser.add_argument("--usefulness-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_training_acceptance(
        args.comparison, args.hpm_report, args.qwen_raw,
        args.correction_report, args.usefulness_report, args.acquisition_report,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
