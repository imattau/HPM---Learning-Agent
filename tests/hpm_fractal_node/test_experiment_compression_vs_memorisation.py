from hpm_fractal_node.experiments.experiment_compression_vs_memorisation import (
    ExperimentConfig,
    run_experiment,
)


def test_ab_composite_emerges():
    summary = run_experiment(ExperimentConfig(ab_steps=20, abc_steps=10))
    assert summary.ab_first_step is not None


def test_abc_composite_emerges_after_ab():
    summary = run_experiment(ExperimentConfig(ab_steps=20, abc_steps=20))
    assert summary.abc_first_step is not None
    assert summary.ab_first_step is None or summary.abc_first_step >= summary.ab_first_step


def test_reuse_rate_nonzero():
    summary = run_experiment(ExperimentConfig(ab_steps=20, abc_steps=20))
    assert summary.ab_reuse_rate > 0.0


def test_node_growth_bounded_after_compression():
    summary = run_experiment(ExperimentConfig(ab_steps=20, abc_steps=20))
    assert summary.node_counts[-1] <= 20
