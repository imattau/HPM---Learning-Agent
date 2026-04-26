from hpm_ai_v4.simulations.hpm_tool_simulation import run_hpm_tool_simulation


def test_run_hpm_tool_simulation_short(tmp_path):
    history = run_hpm_tool_simulation(
        train_episodes=80,
        validation_episodes=30,
        warmup_episodes=10,
        log_every=40,
        train_families=(0, 1, 2),
        validation_families=(3,),
        num_workers=1,
        checkpoint_dir=str(tmp_path),
    )
    assert len(history) >= 2
    phases = {snap["phase"] for snap in history}
    assert "train" in phases
    assert "validation" in phases
    final_val = [snap for snap in history if snap["phase"] == "validation"][-1]
    assert "exact_match" in final_val
    assert "transfer_gain" in final_val
    assert "memory_size" in final_val
    assert (tmp_path / "hpm_tool_library.pkl").exists()
    assert (tmp_path / "hpm_tool_library.reasoner.json").exists()
    assert (tmp_path / "hpm_tool_library.bundle.json").exists()


def test_tool_simulation_preserves_reloadable_polygraph(tmp_path):
    history = run_hpm_tool_simulation(
        train_episodes=40,
        validation_episodes=10,
        warmup_episodes=6,
        log_every=20,
        train_families=(0, 1),
        validation_families=(0,),
        num_workers=1,
        checkpoint_dir=str(tmp_path),
    )
    final_train = [snap for snap in history if snap["phase"] == "train"][-1]
    final_val = [snap for snap in history if snap["phase"] == "validation"][-1]
    assert final_train["memory_size"] > 0
    assert final_val["memory_size"] > 0
    assert final_val["dev_stage"] in {"surface", "local", "relational", "abstract", "generative"}
