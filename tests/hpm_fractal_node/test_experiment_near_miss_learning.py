from hpm_fractal_node.experiments.experiment_near_miss_learning import (
    ExperimentConfig,
    run_experiment,
)


def test_predicts_c_after_training():
    summary = run_experiment(ExperimentConfig(train_steps=20, probe_steps=20))
    assert summary.prediction_accuracy > 0.0


def test_reuses_abc_during_probe():
    summary = run_experiment(ExperimentConfig(train_steps=20, probe_steps=20))
    assert summary.abc_reuse_count > 0


def test_node_growth_bounded():
    summary = run_experiment(ExperimentConfig(train_steps=20, probe_steps=20))
    assert summary.final_node_count <= summary.initial_node_count + 10


def test_creation_not_dominant_when_completion_works():
    summary = run_experiment(ExperimentConfig(train_steps=20, probe_steps=20))
    assert summary.creation_rate < 0.8


def test_residual_trace_length_matches_probe():
    summary = run_experiment(ExperimentConfig(train_steps=10, probe_steps=12))
    assert len(summary.residual_surprise_by_step) == summary.probe_steps
