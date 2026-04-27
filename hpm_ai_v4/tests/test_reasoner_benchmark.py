from hpm_ai_v4.simulations.reasoner_benchmark import run_reasoner_benchmark


def test_run_reasoner_benchmark_short(tmp_path):
    history = run_reasoner_benchmark(
        checkpoint_dir=str(tmp_path),
        num_workers=1,
        use_dict=True,
    )

    assert isinstance(history, list)
    assert len(history) == 8

    families = {snap["family"] for snap in history}
    assert families == {"continue", "repair", "code_dsl", "control"}
    phases = {snap["phase"] for snap in history}
    assert phases == {"adapt", "eval"}

    for snap in history:
        assert "mode_accuracy" in snap
        assert "task_score" in snap
        assert "graph_summary" in snap
        assert "control_context" in snap
        assert "pattern" in snap
        assert "mode_prior" in snap
        assert snap["graph_summary"]["candidate_count"] >= 0
        assert snap["pattern"]["l4_mi"] >= 0.0
        assert snap["pattern"]["l5_mi"] >= 0.0

    assert (tmp_path / "reasoner_benchmark_library.l1.pkl").exists()
    assert (tmp_path / "reasoner_benchmark_library.l2.pkl").exists()
    assert (tmp_path / "reasoner_benchmark_library.l3.pkl").exists()
    assert (tmp_path / "reasoner_benchmark_library.l4.pkl").exists()
    assert (tmp_path / "reasoner_benchmark_library.l5.pkl").exists()
    assert (tmp_path / "reasoner_benchmark_library.mode_prior.json").exists()
