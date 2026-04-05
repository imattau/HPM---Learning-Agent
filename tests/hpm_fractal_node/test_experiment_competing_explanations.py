from hpm_fractal_node.experiments.experiment_competing_explanations import (
    ExperimentConfig,
    run_experiment,
)


def test_early_phase_prefers_cheap_reuse():
    summary = run_experiment()
    assert summary.early_reuse_rate >= 0.70


def test_sustained_ambiguity_triggers_creation():
    summary = run_experiment()
    assert summary.creation_step is not None
    assert summary.creation_step > 0
    assert summary.creation_step < summary.samples


def test_post_creation_error_improves():
    summary = run_experiment()
    assert summary.creation_step is not None
    assert summary.post_creation_error_mean < summary.pre_creation_error_mean


def test_no_duplicate_structure_explosion():
    summary = run_experiment()
    assert summary.created_nodes == 1
    assert not summary.explosion


def test_explicit_stagnation_mode_is_detected():
    summary = run_experiment(
        ExperimentConfig(
            samples=160,
            base_creation_cost=1.40,
            pressure_gain=0.0005,
            min_creation_cost=0.9,
        )
    )
    assert summary.creation_step is None
    assert summary.stagnation
