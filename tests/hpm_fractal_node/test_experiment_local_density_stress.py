from hpm_fractal_node.experiments.experiment_local_density_stress import (
    ExperimentConfig,
    run_experiment,
)


def test_dense_region_still_differentiates():
    summary = run_experiment()
    assert summary.dense_phase.creations >= 1


def test_sparse_creation_at_least_dense():
    summary = run_experiment()
    assert summary.sparse_phase.creations >= 0


def test_density_ratio_higher_in_dense_region():
    summary = run_experiment()
    assert summary.dense_phase.mean_ratio >= summary.sparse_phase.mean_ratio


def test_stagnation_detected_when_suppression_aggressive():
    summary = run_experiment(
        ExperimentConfig(
            residual_surprise_threshold=0.1,
            lacunarity_creation_factor=0.1,
            dense_inputs=6,
            sparse_inputs=2,
            max_new_nodes=1,
        )
    )
    assert summary.stagnation


def test_no_dense_explosion_in_default():
    summary = run_experiment()
    assert not summary.explosion
