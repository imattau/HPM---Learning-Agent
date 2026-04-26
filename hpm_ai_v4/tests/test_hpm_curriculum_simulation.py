from hpm_ai_v4.simulations.hpm_curriculum_simulation import run_hpm_curriculum_simulation


def test_run_hpm_curriculum_simulation_short(tmp_path):
    history = run_hpm_curriculum_simulation(
        train_episodes=90,
        validation_episodes=30,
        warmup_episodes=12,
        episode_length=4,
        log_every=30,
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
    assert "family" in final_val
    assert "transfer_gain" in final_val
    assert "memory_size" in final_val
    assert (tmp_path / "hpm_curriculum_library.pkl").exists()
    assert (tmp_path / "hpm_curriculum_library.reasoner.json").exists()
    assert (tmp_path / "hpm_curriculum_library.bundle.json").exists()


def test_curriculum_simulation_uses_reloadable_policy(tmp_path):
    history = run_hpm_curriculum_simulation(
        train_episodes=60,
        validation_episodes=10,
        warmup_episodes=8,
        episode_length=3,
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
