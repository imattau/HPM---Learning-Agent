import json

import pytest

from hpm_ai_v2.experiments.blinded_usefulness import (
    analyze_ratings,
    build_blinded_packet,
)


def test_blind_packet_excludes_routing_only_items_and_keeps_key_separate(tmp_path):
    curriculum = tmp_path / "curriculum.jsonl"
    rows = [
        {"kind": "source", "source_id": "notes", "title": "Study Notes", "text": "Neurons communicate across synapses."},
        {"kind": "episode", "episode_id": "qa-1", "split": "test", "task_type": "grounded_qa", "question": "How do neurons communicate?", "evidence_id": "notes", "expected_answer": "Neurons communicate across synapses."},
        {"kind": "episode", "episode_id": "unknown-1", "split": "test", "task_type": "abstain", "question": "What is the name of the lecturer?", "evidence_id": None},
        {"kind": "episode", "episode_id": "route-1", "split": "test", "task_type": "procedure", "question": "Summarize the notes.", "evidence_id": None, "workflow_key": "summarize"},
    ]
    curriculum.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    hpm_report = tmp_path / "hpm.json"
    hpm_report.write_text(json.dumps({"final_exam": {"episode_results": [
        {"episode_id": "qa-1", "selected_evidence_id": "notes", "selected_evidence_ids": [], "answer": "Neurons communicate across synapses.", "citations": [], "abstained": False, "answer_token_f1": 1.0},
        {"episode_id": "unknown-1", "selected_evidence_id": None, "selected_evidence_ids": [], "answer": None, "citations": [], "abstained": True},
        {"episode_id": "route-1", "selected_evidence_id": None, "selected_evidence_ids": [], "answer": "summarize", "selected_workflow_key": "summarize", "abstained": False},
    ]}}), encoding="utf-8")
    baseline_report = tmp_path / "baseline.json"
    baseline_report.write_text(json.dumps({"model": "qwen-test", "results": {"episodes": [
        {"episode_id": "qa-1", "selected_evidence_id": "notes", "selected_evidence_ids": [], "answer": "Neurons communicate across synapses.", "task_correct": True, "answer_token_f1": 1.0},
        {"episode_id": "unknown-1", "selected_evidence_id": None, "selected_evidence_ids": [], "answer": "", "abstained": True, "task_correct": True},
        {"episode_id": "route-1", "selected_evidence_id": None, "selected_evidence_ids": [], "answer": "summarize", "selected_workflow_key": "summarize", "task_correct": True},
    ]}}), encoding="utf-8")
    source_state = tmp_path / "sources.json"
    source_state.write_text(json.dumps([{"source_id": "notes", "title": "Study Notes", "text": "Neurons communicate across synapses."}]), encoding="utf-8")
    out = tmp_path / "packet"
    private_key = tmp_path / "private" / "key.json"

    manifest = build_blinded_packet(
        hpm_report, baseline_report, [curriculum], source_state, out, private_key,
        seed=41, sample_size=2,
    )

    packet = [json.loads(line) for line in (out / "rating_packet.jsonl").read_text().splitlines()]
    key = json.loads(private_key.read_text())
    assert manifest["packet_count"] == 2
    assert manifest["excluded_task_counts"] == {"procedure": 1}
    assert manifest["task_counts"] == {"abstain": 1, "grounded_qa": 1}
    assert "HPM" not in json.dumps(packet)
    assert "qwen-test" not in json.dumps(packet)
    grounded_case = next(row for row in packet if row["task_type"] == "grounded_qa")
    assert grounded_case["response_a"]["sources"][0]["title"] == "Study Notes"
    case_key = {row["case_id"]: row for row in key["case_mapping"]}
    assert next(row for row in case_key.values() if row["episode_id"] == "qa-1")
    assert {key["case_mapping"][0]["response_a_system"], key["case_mapping"][0]["response_b_system"]} == {"HPM", "qwen-test"}
    assert (out / "ratings_template.jsonl").exists()
    assert (out / "ratings_template.csv").exists()
    assert (out / "rating_instructions.md").exists()
    assert not (out / "private_deblinding_key.json").exists()


def test_blind_packet_accepts_cumulative_checkpoint_report_schema(tmp_path):
    curriculum = tmp_path / "curriculum.jsonl"
    source = {"kind": "source", "source_id": "notes", "title": "Notes", "text": "Neurons signal across synapses."}
    episode = {"kind": "episode", "episode_id": "qa-new", "split": "test", "task_type": "grounded_qa", "question": "How do neurons signal?", "evidence_id": "notes", "expected_answer": "Neurons signal across synapses."}
    curriculum.write_text(json.dumps(source) + "\n" + json.dumps(episode) + "\n", encoding="utf-8")
    hpm = tmp_path / "hpm.json"
    hpm.write_text(json.dumps({"retention_after_all_phases": {"episode_results": [
        {"episode_id": "qa-new", "task_type": "grounded_qa", "selected_evidence_id": "notes", "selected_evidence_ids": [], "answer": "Neurons signal across synapses.", "citations": [], "abstained": False, "answer_token_f1": 1.0},
    ]}}), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"ollama_retrieval_baseline": {
        "model": "qwen3.5:latest", "episodes": [
            {"episode_id": "qa-new", "task_type": "grounded_qa", "selected_evidence_id": "notes", "selected_evidence_ids": [], "answer": "Neurons signal across synapses.", "task_correct": True, "answer_token_f1": 1.0},
        ],
    }}), encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"sources": [{"source_id": "notes", "title": "Notes", "text": "Neurons signal across synapses."}]}), encoding="utf-8")

    manifest = build_blinded_packet(
        hpm, baseline, [curriculum], state, tmp_path / "packet", tmp_path / "private-key.json",
    )

    assert manifest["packet_count"] == 1
    key = json.loads((tmp_path / "private-key.json").read_text())
    assert key["baseline_model"] == "qwen3.5:latest"


def test_blinded_rating_analysis_maps_labels_and_applies_per_rater_gate(tmp_path):
    packet = tmp_path / "packet.jsonl"
    packet.write_text(json.dumps({"case_id": "case-0001", "task_type": "grounded_qa"}) + "\n", encoding="utf-8")
    key = tmp_path / "key.json"
    key.write_text(json.dumps({
        "hpm_model": "HPM",
        "baseline_model": "Qwen 9B",
        "case_mapping": [{
            "case_id": "case-0001",
            "response_a_system": "HPM",
            "response_b_system": "Qwen 9B",
        }],
    }), encoding="utf-8")
    ratings = tmp_path / "ratings.csv"
    ratings.write_text(
        "case_id,rater_id,usefulness_a,usefulness_b,preferred,comment\n"
        "case-0001,rater-1,5,3,A,\n",
        encoding="utf-8",
    )

    result = analyze_ratings(packet, ratings, key, min_ratings_per_rater=1)

    assert result["per_system"]["HPM"]["useful_or_very_useful_rate"] == 1.0
    assert result["per_system"]["Qwen 9B"]["useful_or_very_useful_rate"] == 0.0
    assert result["per_task_type"]["grounded_qa"]["HPM"]["ratings"] == 1
    assert result["pairwise_preference"] == {"hpm_wins": 1, "baseline_wins": 0, "ties": 0}
    assert result["user_acceptance_gate"]["passed"] is True
    assert len(result["user_acceptance_gate"]["passing_user_rate_wilson_ci95"]) == 2
    assert result["user_acceptance_gate"]["user_level_summary"]["mean_hpm_minus_baseline_cluster_bootstrap_ci95"] is None


def test_blinded_rating_analysis_rejects_duplicate_rater_case(tmp_path):
    packet = tmp_path / "packet.jsonl"
    packet.write_text(json.dumps({"case_id": "case-1", "task_type": "grounded_qa"}) + "\n", encoding="utf-8")
    key = tmp_path / "key.json"
    key.write_text(json.dumps({"hpm_model": "HPM", "baseline_model": "9B", "case_mapping": [
        {"case_id": "case-1", "response_a_system": "HPM", "response_b_system": "9B"}
    ]}), encoding="utf-8")
    ratings = tmp_path / "ratings.jsonl"
    row = {"case_id": "case-1", "rater_id": "same", "usefulness_a": 4, "usefulness_b": 4, "preferred": "tie"}
    ratings.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate rating"):
        analyze_ratings(packet, ratings, key)
