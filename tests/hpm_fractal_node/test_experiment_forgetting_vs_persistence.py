from hpm_fractal_node.experiments.experiment_forgetting_vs_persistence import (
    ExperimentConfig,
    run_experiment,
)


def test_forgetting_vs_persistence_basic_shape():
    summary = run_experiment(ExperimentConfig(abc_steps=8, xyz_steps=8, replay_steps=4))

    assert summary.abc_first_step is not None
    assert summary.xyz_first_step is not None
    assert summary.abc_weight_end_phase2 < summary.abc_weight_end_phase1
    assert summary.xyz_weight_end_phase2 > 0.0
    assert summary.total_pruned > 0
    assert not summary.catastrophic_interference
