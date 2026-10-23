import json

from hpm_ai_v2.experiments.audit_training_acceptance import audit_training_acceptance


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_acceptance_audit_keeps_human_and_correction_safety_gates_explicit(tmp_path):
    families = {
        "grounded_qa": {"total": 60, "accuracy": 0.9},
        "correction": {"total": 60, "accuracy": 0.9},
        "compare": {"total": 60, "accuracy": 0.9},
        "preference": {"total": 60, "accuracy": 0.9},
        "abstain": {"total": 60, "accuracy": 0.9},
        "procedure": {"total": 60, "accuracy": 0.9},
    }
    comparison = {
        "benchmark": "test",
        "episode_count": 360,
        "protocol": {"ollama_model": "qwen-9b", "parameter_size": "9B"},
        "hpm": {
            "accuracy": 0.9,
            "grounded_answer_accuracy": 0.9,
            "answer_token_f1": 0.9,
            "evidence_accuracy": 0.9,
            "median_latency_ms": 1.0,
            "per_task_type": families,
        },
        "qwen_bm25": {
            "accuracy": 0.85,
            "grounded_answer_accuracy": 0.85,
            "answer_token_f1": 0.91,
            "evidence_accuracy": 0.8,
            "median_latency_ms": 1000.0,
            "per_task_type": {key: {"accuracy": 0.85} for key in families},
        },
    }
    hpm = {
        "all_retention_gates_passed": True,
        "final_exam": {"episode_results": [
            {"task_type": "abstain", "abstained": True},
            {"task_type": "abstain", "abstained": True},
        ]},
    }
    qwen_raw = {"results": {"episodes": [
        {"task_type": "abstain", "abstained": True},
        {"task_type": "abstain", "abstained": True},
    ]}}
    unchanged = {"episode_id": "case-collateral-1", "answer": "same", "selected_evidence_id": "source", "answer_token_f1": 1.0}
    changed = {**unchanged, "answer": "changed", "answer_token_f1": 0.0}
    correction = {"training": {"phases": [{
        "phase_test_before_metrics": {"episode_results": [unchanged]},
        "phase_test_metrics": {
            "per_task_type": {"correction": {"accuracy": 0.9, "total": 10}},
            "episode_results": [changed],
        },
    }]}}

    paths = [
        _write(tmp_path / "comparison.json", comparison),
        _write(tmp_path / "hpm.json", hpm),
        _write(tmp_path / "qwen.json", qwen_raw),
        _write(tmp_path / "correction.json", correction),
    ]
    result = audit_training_acceptance(*paths)

    assert result["gates"]["episode_and_family_coverage"]["status"] == "pass"
    assert result["gates"]["task_completion_vs_9b"]["status"] == "pass"
    assert result["gates"]["grounded_answer_quality"]["status"] == "mixed"
    assert result["gates"]["unsupported_answer_rejection"]["status"] == "pass"
    assert result["gates"]["correction_transfer_and_safety"]["status"] == "fail"
    assert result["gates"]["correction_transfer_and_safety"]["fact_validation_gate_passed"] is None
    assert result["gates"]["validated_knowledge_growth"]["status"] == "unmeasured"
    assert result["gates"]["blinded_human_usefulness"]["status"] == "unmeasured"
    assert result["gates"]["latency"]["status"] == "mixed"
    assert result["hpm_configuration"]["answer_generation"].startswith("extractive")
    assert result["acceptance_passed"] is False


def test_correction_gate_requires_held_out_fact_validation(tmp_path):
    correction = {"training": {"phases": [{
        "fit": {"fact_validation_gate": {
            "passed": False, "validated_training_fact_count": 3, "training_fact_count": 8,
        }},
        "phase_test_before_metrics": {"episode_results": [
            {"episode_id": "case-collateral-1", "answer": "same", "selected_evidence_id": "s", "answer_token_f1": 1.0},
        ]},
        "phase_test_metrics": {
            "per_task_type": {"correction": {"accuracy": 0.9, "total": 10}},
            "episode_results": [
                {"episode_id": "case-collateral-1", "answer": "same", "selected_evidence_id": "s", "answer_token_f1": 1.0},
            ],
        },
    }]}}
    comparison = {
        "hpm": {"accuracy": 0.9, "grounded_answer_accuracy": 0.9, "answer_token_f1": 0.9,
                "evidence_accuracy": 0.9, "median_latency_ms": 1.0,
                "per_task_type": {"correction": {"total": 10, "accuracy": 0.9}}},
        "qwen_bm25": {"accuracy": 0.9, "grounded_answer_accuracy": 0.9, "answer_token_f1": 0.9,
                      "evidence_accuracy": 0.9, "median_latency_ms": 10.0,
                      "per_task_type": {"correction": {"accuracy": 0.9}}},
    }
    result = audit_training_acceptance(
        _write(tmp_path / "comparison.json", comparison),
        _write(tmp_path / "hpm.json", {}), None,
        _write(tmp_path / "correction.json", correction),
    )
    gate = result["gates"]["correction_transfer_and_safety"]
    assert gate["status"] == "fail"
    assert gate["fact_validation_validated"] == 3


def test_knowledge_growth_gate_requires_validated_facts_and_restart(tmp_path):
    comparison = {
        "hpm": {"accuracy": 0.9, "grounded_answer_accuracy": 0.9, "answer_token_f1": 0.9,
                "evidence_accuracy": 0.9, "median_latency_ms": 1.0, "per_task_type": {}},
        "qwen_bm25": {"accuracy": 0.9, "grounded_answer_accuracy": 0.9, "answer_token_f1": 0.9,
                      "evidence_accuracy": 0.9, "median_latency_ms": 10.0, "per_task_type": {}},
    }
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write(checkpoint / "study_training_state.json", {"fact_node_data": {
        "fact-1": {"status": "validated"}, "fact-2": {"status": "promoted"},
    }})
    acquisition = {"checkpoint": str(checkpoint / "study_training_state.json"), "training": {"phases": [{
        "name": "fact-acquisition",
        "fit": {"fact_validation_gate": {
            "passed": True, "training_fact_count": 2, "validated_training_fact_count": 2,
        }, "knowledge_growth": {"new_fact_nodes": 2}},
        "phase_test_before_metrics": {"accuracy": 0.25},
        "phase_test_metrics": {"accuracy": 0.75},
        "phase_test_accuracy_delta": 0.5,
    }, {
        "name": "later-workflow-phase",
        "fit": {"fact_validation_gate": {
            "passed": None, "training_fact_count": 0, "validated_training_fact_count": 0,
        }, "knowledge_growth": {"new_fact_nodes": 0}},
    }], "all_retention_gates_passed": True},
        "restart_consistency": {"same_predictions": True}}
    cumulative_hpm = {
        "training": {"all_retention_gates_passed": True},
        "retention_after_all_phases": {"episode_results": []},
    }
    result = audit_training_acceptance(
        _write(tmp_path / "comparison.json", comparison),
        _write(tmp_path / "hpm.json", cumulative_hpm), None, None, None,
        _write(tmp_path / "acquisition.json", acquisition),
    )
    gate = result["gates"]["validated_knowledge_growth"]
    assert gate["status"] == "pass"
    assert gate["new_fact_nodes"] == 2
    assert gate["checkpoint_validated_fact_nodes"] == 2
    assert gate["test_accuracy_delta"] == 0.5
    assert result["gates"]["knowledge_transfer_effect"]["status"] == "pass"
    assert result["gates"]["retention"]["status"] == "pass"
    acquisition["restart_consistency"]["same_predictions"] = False
    result = audit_training_acceptance(
        _write(tmp_path / "comparison.json", comparison),
        _write(tmp_path / "hpm.json", cumulative_hpm), None, None, None,
        _write(tmp_path / "acquisition.json", acquisition),
    )
    assert result["gates"]["validated_knowledge_growth"]["status"] == "fail"
    acquisition["restart_consistency"]["same_predictions"] = True
    acquisition["training"]["phases"][0]["phase_test_accuracy_delta"] = 0.0
    result = audit_training_acceptance(
        _write(tmp_path / "comparison.json", comparison),
        _write(tmp_path / "hpm.json", cumulative_hpm), None, None, None,
        _write(tmp_path / "acquisition.json", acquisition),
    )
    assert result["gates"]["validated_knowledge_growth"]["status"] == "pass"
    assert result["gates"]["knowledge_transfer_effect"]["status"] == "mixed"
    acquisition["restart_consistency"]["same_predictions"] = True
    _write(checkpoint / "study_training_state.json", {"fact_node_data": {
        "fact-1": {"status": "candidate"}, "fact-2": {"status": "candidate"},
    }})
    result = audit_training_acceptance(
        _write(tmp_path / "comparison.json", comparison),
        _write(tmp_path / "hpm.json", cumulative_hpm), None, None, None,
        _write(tmp_path / "acquisition.json", acquisition),
    )
    assert result["gates"]["validated_knowledge_growth"]["status"] == "fail"


def test_acceptance_audit_does_not_mark_missing_ratings_as_a_pass(tmp_path):
    comparison = {
        "episode_count": 360,
        "hpm": {"accuracy": 1.0, "grounded_answer_accuracy": 1.0, "answer_token_f1": 1.0,
                "evidence_accuracy": 1.0, "median_latency_ms": 1.0,
                "per_task_type": {"qa": {"total": 60, "accuracy": 1.0}}},
        "qwen_bm25": {"accuracy": 1.0, "grounded_answer_accuracy": 1.0, "answer_token_f1": 1.0,
                      "evidence_accuracy": 1.0, "median_latency_ms": 10.0,
                      "per_task_type": {"qa": {"accuracy": 1.0}}},
    }
    hpm = {"all_retention_gates_passed": True, "final_exam": {"episode_results": []}}
    result = audit_training_acceptance(
        _write(tmp_path / "comparison.json", comparison),
        _write(tmp_path / "hpm.json", hpm),
        None,
        None,
    )

    assert result["gates"]["blinded_human_usefulness"]["status"] == "unmeasured"
    assert result["acceptance_passed"] is False
