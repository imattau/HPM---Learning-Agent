from hpm_fractal_node.experiments.experiment_absorption_as_generalisation import (
    ExperimentConfig,
    run_experiment,
)


def test_absorption_occurs_under_generalisation_pressure():
    summary = run_experiment(ExperimentConfig(pressure_steps=20))
    assert summary.merge_step is not None
    assert len(summary.absorbed_ids) >= 1


def test_variant_node_count_decreases():
    summary = run_experiment(ExperimentConfig(pressure_steps=20))
    assert summary.final_variant_nodes < summary.initial_variant_nodes


def test_shared_core_preserved():
    summary = run_experiment(ExperimentConfig(pressure_steps=20))
    assert summary.lost_shared_core is False


def test_variant_axis_variance_exceeds_shared_axis():
    summary = run_experiment(ExperimentConfig(pressure_steps=20))
    assert summary.post_variant_sigma_mean > summary.post_shared_sigma_mean


def test_reuse_improves_after_absorption():
    summary = run_experiment(ExperimentConfig(pressure_steps=20))
    assert summary.post_probe_reuse_rate >= summary.pre_probe_reuse_rate


def test_no_unexpected_leaf_creation():
    summary = run_experiment(ExperimentConfig(pressure_steps=20))
    assert summary.created_leaf_nodes == 0
