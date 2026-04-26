from hpm_ai_v4.simulations.hpm_environment_simulation import run_hpm_environment_simulation


def test_run_hpm_environment_simulation_short(tmp_path):
    history = run_hpm_environment_simulation(
        train_steps=120,
        validation_steps=60,
        warmup_steps=20,
        log_every=60,
        state_dim=6,
        train_family=0,
        validation_family=1,
        num_workers=1,
        checkpoint_dir=str(tmp_path),
    )
    assert len(history) >= 2
    phases = {snap["phase"] for snap in history}
    assert "train" in phases
    assert "validation" in phases
    final_val = [snap for snap in history if snap["phase"] == "validation"][-1]
    assert "action_accuracy" in final_val
    assert "transfer_gain" in final_val
    assert "memory_size" in final_val
    assert (tmp_path / "hpm_environment_library.pkl").exists()
    assert (tmp_path / "hpm_environment_library.reasoner.json").exists()
    assert (tmp_path / "hpm_environment_library.bundle.json").exists()


def test_environment_simulation_reports_reloadable_state(tmp_path):
    history = run_hpm_environment_simulation(
        train_steps=60,
        validation_steps=20,
        warmup_steps=10,
        log_every=40,
        state_dim=4,
        train_family=0,
        validation_family=0,
        num_workers=1,
        checkpoint_dir=str(tmp_path),
    )
    final_train = [snap for snap in history if snap["phase"] == "train"][-1]
    final_val = [snap for snap in history if snap["phase"] == "validation"][-1]
    assert final_train["memory_size"] > 0
    assert final_val["memory_size"] > 0
    assert final_val["dev_stage"] in {"surface", "local", "relational", "abstract", "generative"}
